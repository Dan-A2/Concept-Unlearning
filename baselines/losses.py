import torch
import torch.nn as nn
import torch.nn.functional as F


def make_labels(input_ids, attention_mask):
    return input_ids.clone().long().masked_fill(attention_mask == 0, -100)


def compute_loss_from_logits(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    # Mask any label IDs that fall outside the actual vocab dim (e.g. special tokens
    # added by instruction fine-tuning whose IDs exceed config.vocab_size).
    shift_labels = shift_labels.masked_fill(shift_labels >= shift_logits.size(-1), -100)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def ihl(updated_model, unlearn_inputs):
    """Inverted Hinge Loss: bounded unlearning loss that pushes the correct
    token probability below the strongest competitor."""
    labels = make_labels(unlearn_inputs.input_ids, unlearn_inputs.attention_mask)
    outputs = updated_model(**unlearn_inputs)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

    # Flatten to [N, vocab]
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    # Keep only non-padding positions
    mask = flat_labels != -100
    # Also clamp out-of-range labels
    mask = mask & (flat_labels >= 0) & (flat_labels < flat_logits.size(-1))
    flat_logits = flat_logits[mask]
    flat_labels = flat_labels[mask]

    if flat_labels.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    probs = flat_logits.softmax(dim=-1)

    # Probability of correct token
    correct_probs = probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)

    # Max probability among incorrect tokens
    one_hot = F.one_hot(flat_labels, num_classes=probs.size(-1)).bool()
    masked_probs = probs.masked_fill(one_hot, -1.0)
    max_other_probs = masked_probs.max(dim=-1).values

    # Inverted hinge: margin = p(correct) - p(best_other)
    # Loss = clamp(1 + margin, min=0)
    margin = correct_probs - max_other_probs
    loss = torch.clamp(1.0 + margin, min=0.0)

    return loss.mean()


def ga(updated_model, unlearn_inputs):
    """Gradient Ascent: negate cross-entropy on forget data."""
    labels = make_labels(unlearn_inputs.input_ids, unlearn_inputs.attention_mask)
    outputs = updated_model(**unlearn_inputs)
    return -compute_loss_from_logits(outputs.logits, labels)


def npo(updated_model, ref_model, unlearn_inputs, beta, nu=0.0):
    labels = make_labels(unlearn_inputs.input_ids, unlearn_inputs.attention_mask)
    unlearn_outputs = updated_model(**unlearn_inputs)
    current_unlearn_loss = compute_loss_from_logits(unlearn_outputs.logits, labels)

    with torch.no_grad():
        ref_outputs = ref_model.forward_with_injected_noise(
            input_ids=unlearn_inputs.input_ids,
            attention_mask=unlearn_inputs.attention_mask,
            nu=nu,
        )
        ref_unlearn_loss = compute_loss_from_logits(ref_outputs.logits, labels)
    ref_unlearn_loss = ref_unlearn_loss.to(current_unlearn_loss.device)

    neg_log_ratios = current_unlearn_loss - ref_unlearn_loss
    loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta
    return loss


def sim_npo(updated_model, unlearn_inputs, beta):
    labels = make_labels(unlearn_inputs.input_ids, unlearn_inputs.attention_mask)
    unlearn_outputs = updated_model(**unlearn_inputs)
    current_unlearn_loss = compute_loss_from_logits(unlearn_outputs.logits, labels)

    loss = -F.logsigmoid(beta * current_unlearn_loss).mean() * 2 / beta
    return loss


def dpo(updated_model, ref_model, unlearn_inputs, idk_inputs, beta, nu=0.0):
    unlearn_labels = make_labels(unlearn_inputs.input_ids, unlearn_inputs.attention_mask)
    idk_labels = make_labels(idk_inputs.input_ids, idk_inputs.attention_mask)

    unlearn_outputs = updated_model(**unlearn_inputs)
    idk_outputs = updated_model(**idk_inputs)

    idk_current_loss = -1.0 * compute_loss_from_logits(idk_outputs.logits, idk_labels)
    unlearn_current_loss = -1.0 * compute_loss_from_logits(unlearn_outputs.logits, unlearn_labels)

    with torch.no_grad():
        idk_ref_outputs = ref_model.forward_with_injected_noise(
            input_ids=idk_inputs.input_ids,
            attention_mask=idk_inputs.attention_mask,
            nu=nu,
        )
        unlearn_ref_outputs = ref_model.forward_with_injected_noise(
            input_ids=unlearn_inputs.input_ids,
            attention_mask=unlearn_inputs.attention_mask,
            nu=nu,
        )
        idk_ref_loss = -1.0 * compute_loss_from_logits(idk_ref_outputs.logits, idk_labels)
        unlearn_ref_loss = -1.0 * compute_loss_from_logits(unlearn_ref_outputs.logits, unlearn_labels)

    pi_log_ratios = idk_current_loss - unlearn_current_loss
    ref_log_ratios = (idk_ref_loss - unlearn_ref_loss).to(pi_log_ratios.device)

    loss = -F.logsigmoid(beta * (pi_log_ratios - ref_log_ratios)).mean() * 2 / beta
    return loss


def mse(updated_model, ref_model, retain_inputs, nu):
    retain_outputs = updated_model(**retain_inputs)
    retain_logits = retain_outputs.logits

    with torch.no_grad():
        ref_outputs = ref_model.forward_with_injected_noise(
            input_ids=retain_inputs.input_ids,
            attention_mask=retain_inputs.attention_mask,
            nu=nu,
        )
    ref_logits = ref_outputs.logits.to(retain_logits.device)

    probs = F.log_softmax(retain_logits, dim=-1).view(-1, retain_logits.shape[-1]).to(torch.bfloat16)
    ref_probs = F.log_softmax(ref_logits, dim=-1).view(-1, ref_logits.shape[-1]).to(torch.bfloat16)

    retain_loss = F.mse_loss(probs, ref_probs, reduction='mean')
    return retain_loss


def kl(updated_model, ref_model, retain_inputs, nu):
    retain_outputs = updated_model(**retain_inputs)
    retain_logits = retain_outputs.logits

    with torch.no_grad():
        ref_outputs = ref_model.forward_with_injected_noise(
            input_ids=retain_inputs.input_ids,
            attention_mask=retain_inputs.attention_mask,
            nu=nu,
        )
    ref_logits = ref_outputs.logits.to(retain_logits.device)

    probs = F.log_softmax(retain_logits.float(), dim=-1).view(-1, retain_logits.shape[-1])
    ref_probs = F.log_softmax(ref_logits.float(), dim=-1).view(-1, ref_logits.shape[-1])
    retain_loss = F.kl_div(probs, ref_probs, reduction='batchmean', log_target=True)
    return retain_loss


def kl_frozen(updated_model, frozen_model, retain_inputs):
    """KL divergence between updated model and a plain frozen model (no noise)."""
    retain_outputs = updated_model(**retain_inputs)
    retain_logits = retain_outputs.logits

    with torch.no_grad():
        ref_outputs = frozen_model(**retain_inputs)
    ref_logits = ref_outputs.logits.to(retain_logits.device)

    probs = F.log_softmax(retain_logits.float(), dim=-1).view(-1, retain_logits.shape[-1])
    ref_probs = F.log_softmax(ref_logits.float(), dim=-1).view(-1, ref_logits.shape[-1])
    return F.kl_div(probs, ref_probs, reduction='batchmean', log_target=True)


def obliviate_vocab_kl(updated_model, unlearn_inputs, sensitive_token_ids):
    """Vocabulary-masking KL divergence on forget data (Obliviate).

    Zeroes out sensitive-token logits, then minimises KL between the full
    distribution and the masked distribution.  This pushes the model to
    redistribute probability mass away from sensitive tokens.
    """
    outputs = updated_model(**unlearn_inputs)
    logits = outputs.logits

    vocab_mask = torch.ones(logits.size(-1), device=logits.device)
    for token_id in sensitive_token_ids:
        if token_id < vocab_mask.size(0):
            vocab_mask[token_id] = 0

    logits_masked = logits * vocab_mask
    logits_probs = F.log_softmax(logits, dim=-1)
    target_probs = F.softmax(logits_masked, dim=-1)

    return F.kl_div(
        logits_probs.view(-1, logits.size(-1)),
        target_probs.view(-1, logits.size(-1)),
        reduction="batchmean",
    )


def obliviate_distill_mse(updated_model, frozen_model, retain_inputs):
    """MSE distillation loss between student and teacher logits (Obliviate)."""
    outputs = updated_model(**retain_inputs)
    logits = outputs.logits

    with torch.no_grad():
        teacher_outputs = frozen_model(**retain_inputs)
    teacher_logits = teacher_outputs.logits.to(logits.device)

    return F.mse_loss(
        logits.view(-1, logits.size(-1)),
        teacher_logits.view(-1, teacher_logits.size(-1)),
    )


def obliviate_retain_ce(updated_model, frozen_model, retain_inputs, pad_token_id=-100):
    """CE retain loss against teacher model's argmax targets (Obliviate)."""
    with torch.no_grad():
        teacher_outputs = frozen_model(**retain_inputs)
        teacher_targets = teacher_outputs.logits.softmax(dim=-1).argmax(dim=-1)

    outputs = updated_model(**retain_inputs)
    logits = outputs.logits

    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        teacher_targets.view(-1).to(logits.device),
        ignore_index=pad_token_id,
    )
