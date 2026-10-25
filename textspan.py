"""Small pure-Python fallback for pytextspan used by UQAC evaluation.

The Hugging Face token strings passed by UQAC reconstruct the original text
exactly for the supported instruction models, so a cumulative alignment is
both deterministic and sufficient for answer-span projection.
"""


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
