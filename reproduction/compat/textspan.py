"""Pure-Python fallback for the pytextspan functions used by UQAC."""


def get_original_spans(tokens, text):
    spans = []
    cursor = 0
    for token in tokens:
        if text.startswith(token, cursor):
            start = cursor
        else:
            start = text.find(token, cursor)
            if start < 0:
                spans.append([])
                continue
        end = start + len(token)
        spans.append([(start, end)])
        cursor = end
    return spans


def align_spans(spans, source_text, target_text):
    if source_text != target_text:
        raise ValueError("Fallback align_spans only supports identical texts")
    return [[tuple(span)] for span in spans]
