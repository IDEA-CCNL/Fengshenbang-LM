from contextlib import contextmanager
import signal
import torch
from torchsnooper import snoop

# taken from
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except Exception as e:
        signal.alarm(0)
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None


def use_calculator(sample):
    if "<<" not in sample:
        return None

    parts = sample.split("<<")
    remaining = parts[-1]
    if ">>" in remaining:
        return None
    if "=" not in remaining:
        return None
    lhs = remaining.split("=")[0]
    lhs = lhs.replace(",", "")
    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    return eval_with_timeout(lhs)


#  @snoop()
def sample(model, qn, tokenizer, device, sample_len):
    # Inefficient version of calculator sampling -- no batches, doesn't
    # cache activations from previous tokens
    EQUALS_TOKENS = set([28, 796, 47505])

    generated_token_ids = []
    past_key_values = None
    model_kwargs = {}
    toks = tokenizer([qn], padding=False, return_tensors="pt").input_ids.to(device)
    for _ in range(sample_len):
        with torch.no_grad():
            #  toks = tokenizer([qn], padding=False, return_tensors="pt").to(device)
            #  orig_len = toks["input_ids"].shape[1]
            orig_len = toks.size(1)

            model_kwargs["past"] = past_key_values
            out, model_outputs = model.generate(
                #  **toks, max_length=orig_len + 1, pad_token_id=model.config.eos_token_id
                toks, 
                max_length=orig_len + 1, 
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **model_kwargs,
            )
            generated_token_ids.append(out[0][-1])
            toks = out
            text = tokenizer.batch_decode(out)[0]
            past_key_values = model_outputs.past_key_values

            if out[0, -1].item() in EQUALS_TOKENS:
                answer = use_calculator(text)
                if answer is not None:
                    #  print("Triggered calculator, answer", answer)
                    #  text = text + str(answer) + ">>"
                    toks = torch.cat((toks, tokenizer([str(answer) + ">>"], padding=False, return_tensors="pt").input_ids.to(device)), -1)
                    generated_token_ids.extend(tokenizer.convert_tokens_to_ids([str(answer), ">>"]))
                    past_key_values = None

            #  qn = text

    return tokenizer.batch_decode(toks)[0], generated_token_ids
    #  return qn, generated_token_ids

