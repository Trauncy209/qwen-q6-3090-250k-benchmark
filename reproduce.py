"""Portable reproduction of the recorded JSON retrieval/speed fixture. Python stdlib only."""
from pathlib import Path
import argparse,hashlib,json,random,time,urllib.request
parser=argparse.ArgumentParser()
parser.add_argument('--port',type=int,default=8105)
parser.add_argument('--sizes',type=int,nargs='+',default=[131072,250000])
parser.add_argument('--output',type=Path,default=Path('benchmark-output'))
parser.add_argument('--prepare-only',action='store_true')
args=parser.parse_args();PORT=args.port;ROOT=args.output;ROOT.mkdir(parents=True,exist_ok=True)
TOKEN='unused-local'
def api(port, path, data=None, timeout=20, auth=False):
    headers = {'Content-Type': 'application/json'}
    if auth:
        headers['Authorization'] = 'Bearer ' + TOKEN
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=None if data is None else json.dumps(data).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def extract_json(text):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == '{':
            try:
                return decoder.raw_decode(text[i:])[0]
            except ValueError:
                pass
    return None

def tokenize(port, text):
    return api(port, '/tokenize', {'content': text, 'add_special': False}, timeout=90)['tokens']

def render(port, messages):
    return api(port, '/apply-template', {'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}})['prompt']

def prepare_prompts():
    expected = {'amber': 'violetcedar', 'birch': 'cobaltfern', 'cedar': 'silverotter', 'delta': 'goldenbirch'}
    queries = '\nReturn one JSON object mapping amber, birch, cedar and delta to the exact registered string values. Read the authoritative REGISTER lines; ignore ordinary archive records. No explanations.'
    template = render(PORT, [{'role': 'system', 'content': 'You are a precise data extraction assistant. Return the requested JSON only. Do not think aloud.'}, {'role': 'user', 'content': 'ARCHIVE_START\n@@BODY@@\nARCHIVE_END' + queries}])
    prefix, suffix = template.split('@@BODY@@')
    pre = tokenize(PORT, prefix)
    post = tokenize(PORT, suffix)
    rng = random.Random(271828)
    words = ['archive', 'worker', 'pending', 'cached', 'forest', 'sensor', 'request', 'latency', 'branch', 'record', 'session', 'network', 'status', 'verified', 'sample', 'memory']
    filler = '\n'.join(('entry ' + str(i) + ': ' + ' '.join((rng.choice(words) for _ in range(16))) + ' code ' + str(rng.randrange(100000, 999999)) for i in range(18000)))
    fill = tokenize(PORT, filler)
    needles = [tokenize(PORT, '\nREGISTER ' + key + ' = ' + value + '\n') for key, value in expected.items()]
    prompts = {}
    for size in [4096, 32768, 131072, 250000]:
        budget = size - len(pre) - len(post) - sum(map(len, needles))
        assert len(fill) >= budget
        body = []
        used = 0
        for fraction, needle in zip([0.12, 0.43, 0.74, 0.96], needles):
            end = int(budget * fraction)
            body.extend(fill[used:end])
            body.extend(needle)
            used = end
        body.extend(fill[used:budget])
        tokens = pre + body + post
        assert len(tokens) == size
        p = ROOT / f'prompt-{size}.json'
        p.write_text(json.dumps(tokens, separators=(',', ':')))
        prompts[str(size)] = {'file': str(p), 'tokens': size, 'sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(), 'expected': expected}
    (ROOT / 'prompts.json').write_text(json.dumps(prompts, indent=2))
    return prompts

def request_completion(port, prompt, n_predict=128, fixed_output=True):
    body = {'prompt': prompt, 'n_predict': n_predict, 'stream': True, 'temperature': 0, 'seed': 12345, 'cache_prompt': False, 'ignore_eos': fixed_output, 'timings_per_token': False}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/completion', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    first = None
    content = []
    final = None
    with urllib.request.urlopen(req, timeout=3600) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            raw = line[6:].strip()
            if raw == b'[DONE]':
                continue
            data = json.loads(raw)
            if data.get('error'):
                raise RuntimeError(str(data['error']))
            text = data.get('content', '')
            if text:
                if first is None:
                    first = time.monotonic()
                content.append(text)
            if data.get('stop'):
                final = data
    if final is None:
        raise RuntimeError('Completion stream ended without final timings')
    return {'elapsed_s': time.monotonic() - start, 'ttft_s': None if first is None else first - start, 'timings': final.get('timings'), 'tokens_predicted': final.get('tokens_predicted'), 'tokens_evaluated': final.get('tokens_evaluated'), 'truncated': final.get('truncated'), 'stop_type': final.get('stop_type'), 'content': ''.join(content)}

prompts=prepare_prompts()
if not args.prepare_only:
    for size in args.sizes:
        meta=prompts[str(size)];tokens=json.loads(Path(meta['file']).read_text())
        result=request_completion(PORT,tokens,128,True)
        parsed=extract_json(result['content']);expected=meta['expected']
        result.update(target_prompt_tokens=size,retrieval_correct=sum(parsed.get(k)==v for k,v in expected.items())if isinstance(parsed,dict)else 0,retrieval_total=len(expected),prompt_sha256=meta['sha256'])
        with(ROOT/'measurements.jsonl').open('a')as f:f.write(json.dumps(result)+'\n')
        print(json.dumps(result),flush=True)
        assert result['timings']['prompt_n']==size and result['timings']['predicted_n']==128 and not result['truncated']
