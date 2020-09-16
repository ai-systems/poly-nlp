from transformers.tokenization_bert import BertTokenizer


def encode_sentence_pairs(
    sentence1: str,
    sentence2: str = None,
    tokenizer: BertTokenizer = BertTokenizer,
    add_special_tokens=True,
    max_length=512,
    label_list=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    inputs = tokenizer.encode_plus(
        sentence1,
        sentence2,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
    )
    input_ids, token_type_ids = (
        inputs["input_ids"],
        inputs["token_type_ids"],
    )
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = (
            [0 if mask_padding_with_zero else 1] * padding_length
        ) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + (
            [0 if mask_padding_with_zero else 1] * padding_length
        )
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
        len(input_ids), max_length
    )
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
        len(attention_mask), max_length
    )
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
        len(token_type_ids), max_length
    )

    return input_ids, attention_mask, token_type_ids


def convert_token_ids(
    tokens: list,
    tokenizer: BertTokenizer = BertTokenizer,
    add_special_tokens=True,
    max_length=512,
    label_list=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    input_ids = [101] + tokenizer.convert_tokens_to_ids(tokens) + [102]
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = (
            [0 if mask_padding_with_zero else 1] * padding_length
        ) + attention_mask
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + (
            [0 if mask_padding_with_zero else 1] * padding_length
        )

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
        len(input_ids), max_length
    )
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
        len(attention_mask), max_length
    )

    return input_ids, attention_mask
