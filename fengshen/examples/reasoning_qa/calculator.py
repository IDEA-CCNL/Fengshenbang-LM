from contextlib import contextmanager
import signal
import torch as th
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


def batch_calculator_sample_with_early_stop(model, qn, tokenizer, device, sample_len, **kwargs):
    EQUALS_TOKENS = set([28, 796, 47505])
    ANS_TOKEN = set([tokenizer.convert_tokens_to_ids("[ANS]")])
    END_TOKEN = set(tokenizer.convert_tokens_to_ids([tokenizer.eos_token, "\n", "\n\n"]))

    model_kwargs = {}
    past_key_values = None
    generated_token_ids = [[] for _ in range(len(qn))]
    # TODO 假设一个sample生成 [ANS] 后再生成了 <|endoftext|> 或 \n 就代表结束
    ans_generated = [False] * len(qn)
    finished_sample = [""] * len(qn)
    qn_idx_list = range(len(qn))
    tokenizer.padding_side = "left"
    for _ in range(sample_len):
        with th.no_grad():
            inputs_encoding = tokenizer(
                qn,
                return_attention_mask=True,
                return_tensors="pt", 
                add_special_tokens=False,
                padding=True,
            ).to(device)
            #  attention_mask = th.where(inputs_encoding["input_ids"] == tokenizer.pad_token_id, 0, 1)
            #  inputs_encoding["attention_mask"] = attention_mask
            #  inputs_encoding = inputs_encoding.to(device)
            #  if _ == 0 or _ == sample_len - 1:
            #      print("inputs_encoding", inputs_encoding)
            orig_len = inputs_encoding["input_ids"].shape[1]

            if past_key_values and past_key_values[0][0].size(-2) != orig_len - 1:
                #  print("past key values size: ", past_key_values[0][0].size(-2))
                #  print("current input ids length: ", orig_len)
                past_key_values = None

            model_kwargs["past"] = past_key_values
            out, model_outputs = model.generate(
                **inputs_encoding,
                max_length=orig_len + 1,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **model_kwargs,
                **kwargs,
            )
            past_key_values = model_outputs.past_key_values
            text = tokenizer.batch_decode(out, skip_special_tokens=True)
            #  if _ == 0 or _ == sample_len - 1:
                #  print("out", out)
                #  print("text", text)
            for i in range(len(qn)):
                generated_token_ids[qn_idx_list[i]].append(out[i, -1].item())
                #  torch tensor的索引非常慢
                #  if out[i, -1].item() in EQUALS_TOKENS:
                #TODO
                if generated_token_ids[qn_idx_list[i]][-1] in EQUALS_TOKENS:
                    answer = use_calculator(text[i])
                    if answer is not None:
                        #  print("Triggered calculator, answer", answer)
                        text[i] = text[i] + str(answer) + ">>"
                        generated_token_ids[qn_idx_list[i]].extend(tokenizer.convert_tokens_to_ids([str(answer), ">>"]))
                        past_key_values = None

                if generated_token_ids[qn_idx_list[i]][-1] in ANS_TOKEN:
                    ans_generated[qn_idx_list[i]] = True

                if ans_generated[qn_idx_list[i]] and generated_token_ids[qn_idx_list[i]][-1] in END_TOKEN:
                    finished_sample[qn_idx_list[i]] = text[i]
                    qn_idx_list.pop(i)
                    text.pop(i)
            #  因为一开始padding到了同长度，中间生成过程中如果原本短的那个突然因为calculator变长了，
            #  那么长的那个在下一轮也会被pad，所以会出现batch内最长的句子也在开头被padding的情况，所以要在decode出text时skip_special_tokens
            qn = text

            if all(ans_generated):
                current_patience += 1
            if current_patience >= patience_after_all_finished:
                break

    tokenizer.padding_side = "right"

    return qn, generated_token_ids


#  @snoop()
def batch_calculator_sample(model, qn, tokenizer, device, sample_len, **kwargs):
    EQUALS_TOKENS = set([28, 796, 47505])
    ANS_TOKEN = set([tokenizer.convert_tokens_to_ids("[ANS]")])

    model_kwargs = {}
    past_key_values = None
    generated_token_ids = [[] for _ in range(len(qn))]
    finished = [False] * len(qn)
    # TODO 这里假设所有sample都生成 [ANS] 后要生成的答案数字最多只有10个字符和一个<|endoftext|>
    current_patience = 0
    patience_after_all_finished = 11
    tokenizer.padding_side = "left"
    for _ in range(sample_len):
        with th.no_grad():
            inputs_encoding = tokenizer(
                qn,
                return_attention_mask=True,
                return_tensors="pt", 
                add_special_tokens=False,
                padding=True,
            ).to(device)
            #  attention_mask = th.where(inputs_encoding["input_ids"] == tokenizer.pad_token_id, 0, 1)
            #  inputs_encoding["attention_mask"] = attention_mask
            #  inputs_encoding = inputs_encoding.to(device)
            #  if _ == 0 or _ == sample_len - 1:
            #      print("inputs_encoding", inputs_encoding)
            orig_len = inputs_encoding["input_ids"].shape[1]

            if past_key_values and past_key_values[0][0].size(-2) != orig_len - 1:
                #  print("past key values size: ", past_key_values[0][0].size(-2))
                #  print("current input ids length: ", orig_len)
                past_key_values = None

            model_kwargs["past"] = past_key_values
            out, model_outputs = model.generate(
                **inputs_encoding,
                max_length=orig_len + 1,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **model_kwargs,
                **kwargs,
            )
            past_key_values = model_outputs.past_key_values
            text = tokenizer.batch_decode(out, skip_special_tokens=True)
            #  if _ == 0 or _ == sample_len - 1:
                #  print("out", out)
                #  print("text", text)
            for i in range(len(generated_token_ids)):
                generated_token_ids[i].append(out[i, -1].item())
                #  torch tensor的索引非常慢
                #  if out[i, -1].item() in EQUALS_TOKENS:
                if generated_token_ids[i][-1] in EQUALS_TOKENS:
                    answer = use_calculator(text[i])
                    if answer is not None:
                        #  print("Triggered calculator, answer", answer)
                        text[i] = text[i] + str(answer) + ">>"
                        generated_token_ids[i].extend(tokenizer.convert_tokens_to_ids([str(answer), ">>"]))
                        past_key_values = None
                if generated_token_ids[i][-1] in ANS_TOKEN:
                    finished[i] = True

            #  因为一开始padding到了同长度，中间生成过程中如果原本短的那个突然因为calculator变长了，
            #  那么长的那个在下一轮也会被pad，所以会出现batch内最长的句子也在开头被padding的情况，所以要在decode出text时skip_special_tokens
            qn = text

            if all(finished):
                current_patience += 1
            if current_patience >= patience_after_all_finished:
                break

    tokenizer.padding_side = "right"

    return qn, generated_token_ids


def sample(model, qn, tokenizer, device, sample_len):
    # Inefficient version of calculator sampling -- no batches, doesn't
    # cache activations from previous tokens
    EQUALS_TOKENS = set([28, 796, 47505])

    generated_token_ids = []
    past_key_values = None
    model_kwargs = {}
    for _ in range(sample_len):
        with th.no_grad():
            toks = tokenizer([qn], padding=False, return_tensors="pt").to(device)
            orig_len = toks["input_ids"].shape[1]
            if past_key_values and past_key_values[0][0].size(-2) != orig_len - 1:
                past_key_values = None

            model_kwargs["past"] = past_key_values
            out, model_outputs = model.generate(
                #  **toks, max_length=orig_len + 1, pad_token_id=model.config.eos_token_id
                **toks, 
                max_length=orig_len + 1, 
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **model_kwargs,
            )
            generated_token_ids.append(out[0][-1])
            text = tokenizer.batch_decode(out)[0]
            # ([past_key_layer0, past_value_layer0], [past_key_layer1, past_value_layer1], ..., [past_key_layer11, past_value_layer11])
            # len(past_key_values) = num_layers, len(past_key_values[0]) = 2
            # past_key_values[0][0].size() = (batch_size, num_heads, seq_len, hidden_size // num_heads)
            past_key_values = model_outputs.past_key_values

            if out[0, -1].item() in EQUALS_TOKENS:
                answer = use_calculator(text)
                if answer is not None:
                    #  print("Triggered calculator, answer", answer)
                    text = text + str(answer) + ">>"
                    generated_token_ids.extend(tokenizer.convert_tokens_to_ids([str(answer), ">>"]))
                    past_key_values = None

            qn = text
    return qn, generated_token_ids

if __name__ == "__main__":
    text = "He spent $1.5 on soda because 2-.50 = <<2-.50="
    print(use_calculator(text))

