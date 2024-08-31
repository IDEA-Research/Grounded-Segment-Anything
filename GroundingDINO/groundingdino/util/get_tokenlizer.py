from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast


def get_tokenlizer(text_encoder_type, bert_base_uncased_path):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    
    # solve huggingface connect issue
    if is_bert_model_use_local_path(bert_base_uncased_path) and text_encoder_type == "bert-base-uncased":
        print("use local bert model path: {}".format(bert_base_uncased_path))
        return AutoTokenizer.from_pretrained(bert_base_uncased_path)

    print("final text_encoder_type: {}".format(text_encoder_type))

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type)
    return tokenizer


def get_pretrained_language_model(text_encoder_type, bert_base_uncased_path):
    if text_encoder_type == "bert-base-uncased":
        if is_bert_model_use_local_path(bert_base_uncased_path):
            return BertModel.from_pretrained(bert_base_uncased_path)
        return BertModel.from_pretrained(text_encoder_type)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)
    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))

def is_bert_model_use_local_path(bert_base_uncased_path):
    return bert_base_uncased_path is not None and len(bert_base_uncased_path) > 0
