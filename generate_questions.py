# -*- coding: utf-8 -*-
"""
generate_questions.py — авто-генерация блоков «## Возможные вопросы на собеседовании»
для заметок vault. Локально через Ollama (qwen). Разово, на реиндексе / при установке.

Блок вопросов подмешивается в эмбеддинг заметки и резко улучшает попадание e5
(query-shaped якоря: заметка написана утвердительно, а запрос — вопросом).

Идемпотентно: пропускает заметки, где блок уже есть, и слишком короткие.
По умолчанию DRY-RUN (только показывает). Запись — с флагом --apply.

Запуск:
    .venv\\Scripts\\python.exe generate_questions.py [--apply] [--limit N] [--model ...]
"""
import sys, glob, re, time, argparse
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import requests

HEADER       = "## Возможные вопросы на собеседовании"
MARKER_START = "<!-- AUTO_GENERATED_QUERIES_START -->"
MARKER_END   = "<!-- AUTO_GENERATED_QUERIES_END -->"

SYS = (
    "Ты составляешь блок «Возможные вопросы на собеседовании» для DevOps-заметки. "
    "По СОДЕРЖИМОМУ заметки придумай {n} коротких формулировок вопроса, на которые "
    "эта заметка отвечает. Разнообразь форму: формальные и разговорные, с "
    "транслитерацией английских терминов (как реально спрашивают вслух). "
    "Каждый вопрос — ПОЛНОЕ предложение-вопрос, как его задают вслух; НЕ используй "
    "сокращения, стрелки '→', формулы или обрывки (плохо: '137 → что значит?'; "
    "хорошо: 'Что значит код выхода 137?'). "
    "По одной на строку, начиная с '- '. Только вопросы, без ответов и пояснений."
)

RE_HTML  = re.compile(r"<!--.*?-->", re.DOTALL)
RE_IMG   = re.compile(r"!\[\[[^\]]*\]\]")
RE_QSEC  = re.compile(r"\n#+\s*Возможные вопросы.*$", re.DOTALL)
RE_WS    = re.compile(r"[ \t]+")
RE_BULLET = re.compile(r"^[\s\-\*•\d\.\)]+")


def extract_body(raw: str) -> str:
    raw = RE_HTML.sub(" ", raw)
    raw = RE_QSEC.sub("", raw)          # выкинуть уже существующую секцию вопросов
    raw = RE_IMG.sub(" ", raw)
    return RE_WS.sub(" ", raw).strip()


def normalize(text: str, n: int) -> list[str]:
    out, seen = [], set()
    for ln in text.splitlines():
        ln = RE_BULLET.sub("", ln.strip()).strip().strip('"').strip()
        if len(ln) < 6:
            continue
        low = ln.lower()
        # отсечь возможную преамбулу/пояснения модели
        if low.startswith(("вот ", "конечно", "например,", "ниже ", "список")):
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(ln)
    return out[:n]


def generate(url: str, model: str, stem: str, body: str, n: int, timeout: float) -> list[str]:
    resp = requests.post(url, json={
        "model": model,
        "messages": [
            {"role": "system", "content": SYS.format(n=n)},
            {"role": "user", "content": f"Заголовок: {stem}\n\n{body}"},
        ],
        "stream": False,
        "keep_alive": "20m",
        "options": {"temperature": 0.4, "num_predict": 350, "num_gpu": 99},
    }, timeout=timeout)
    resp.raise_for_status()
    return normalize(resp.json()["message"]["content"], n)


def build_block(qs: list[str]) -> str:
    body = "\n".join(f"- {q}" for q in qs)
    return f"\n\n{HEADER}\n{MARKER_START}\n{body}\n{MARKER_END}\n"


def default_folder() -> str:
    """Папка заметок: из data.json (scanFolder) относительно корня волта. Переносимо."""
    here = Path(__file__).resolve()
    vault = here.parents[3]              # …/voice-vault-search → plugins → .obsidian → VAULT
    try:
        import json
        sf = (json.loads((here.parent / "data.json").read_text(encoding="utf-8"))
              .get("scanFolder", "") or "").strip()
        if sf and (vault / sf).exists():
            return str(vault / sf)
    except Exception:
        pass
    return str(vault)                    # весь волт, если scanFolder не задан


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=default_folder())
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--url", default="http://localhost:11434/api/chat")
    ap.add_argument("--num", type=int, default=12, help="сколько вопросов на заметку")
    ap.add_argument("--min-body", type=int, default=120, help="мин. длина тела (символов), иначе пропуск")
    ap.add_argument("--limit", type=int, default=0, help="обработать только первые N (0 = все)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--apply", action="store_true", help="писать в файлы (иначе dry-run)")
    args = ap.parse_args()

    files = [p for p in glob.glob(f"{args.folder}/**/*.md", recursive=True)
             if not Path(p).stem.startswith("_")]
    files.sort()
    if args.limit:
        files = files[:args.limit]

    mode = "APPLY (пишу в файлы)" if args.apply else "DRY-RUN (только показ)"
    print(f"[{mode}] модель={args.model} | заметок={len(files)} | папка={args.folder}\n")

    done = skip_has = skip_short = fail = 0
    t_all = time.perf_counter()
    for i, p in enumerate(files, 1):
        stem = Path(p).stem
        try:
            raw = Path(p).read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[{i}/{len(files)}] ⚠ читать не смог {stem}: {e}"); fail += 1; continue
        if MARKER_START in raw:
            skip_has += 1; print(f"[{i}/{len(files)}] ⏭  {stem} — блок уже есть"); continue
        body = extract_body(raw)
        if len(body) < args.min_body:
            skip_short += 1; print(f"[{i}/{len(files)}] ⏭  {stem} — тело короткое ({len(body)} симв.)"); continue
        try:
            t0 = time.perf_counter()
            qs = generate(args.url, args.model, stem, body, args.num, args.timeout)
            dt = time.perf_counter() - t0
        except Exception as e:
            print(f"[{i}/{len(files)}] ⚠ генерация упала {stem}: {e}"); fail += 1; continue
        if len(qs) < 3:
            print(f"[{i}/{len(files)}] ⚠ мало вопросов ({len(qs)}) для {stem} — пропуск"); fail += 1; continue
        block = build_block(qs)
        done += 1
        print(f"[{i}/{len(files)}] ✓ {stem}  ({dt:.0f}с, {len(qs)} вопр.)")
        if args.apply:
            Path(p).write_text(raw.rstrip() + block, encoding="utf-8")
        else:
            print(block)

    print(f"\n=== итог за {time.perf_counter()-t_all:.0f}с ===")
    print(f"сгенерировано: {done} | уже был блок: {skip_has} | короткие: {skip_short} | ошибки: {fail}")
    if not args.apply and done:
        print("Это DRY-RUN. Записать в файлы — перезапусти с --apply")


if __name__ == "__main__":
    main()
