import os
import math
from typing import List, Dict, Union, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


class SentimentClassifier:
    """
    Обёртка над transformers.pipeline для финансового сентимента.
    Модель по умолчанию: 'mxlcw/rubert-tiny2-russian-financial-sentiment'

    Логика:
      - для title используем батчи (один текст <= 512 токенов)
      - для publication режем на чанки по 512 токенов и агрегируем
    """

    def __init__(self,
                 model_name: str = "mxlcw/rubert-tiny2-russian-financial-sentiment",
                 device: Optional[int] = None,
                 batch_size: int = 16,
                 max_length: int = 512):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=self.tokenizer,
            device=device if device >= 0 else -1,
            return_all_scores=True
        )

    def _postprocess_single(self, scores):
        out = {d["label"]: float(d["score"]) for d in scores}
        best_label = max(out.items(), key=lambda x: x[1])[0]
        return {"label": best_label, "scores": out, "confidence": out[best_label]}

    def _chunk_text(self, text: str, stride: int = 256) -> List[str]:
        """Режет длинный текст на чанки длиной <= max_length с перекрытием."""
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), stride):
            window = tokens[i:i + self.max_length]
            if not window:
                continue
            chunks.append(self.tokenizer.convert_tokens_to_string(window))
            if i + self.max_length >= len(tokens):
                break
        return chunks

    def _aggregate(self, preds: List[dict]) -> dict:
        """Агрегирует предсказания по чанкам (усреднение по score)."""
        if not preds:
            return {}
        labels = preds[0]["scores"].keys()
        agg_scores = {lab: 0.0 for lab in labels}
        for p in preds:
            for lab, val in p["scores"].items():
                agg_scores[lab] += val
        for lab in agg_scores:
            agg_scores[lab] /= len(preds)
        best_label = max(agg_scores.items(), key=lambda x: x[1])[0]
        return {"label": best_label, "scores": agg_scores, "confidence": agg_scores[best_label]}

    def predict(self, text: str, mode: str = "title", meta: Optional[dict] = None) -> dict:
        """
        mode="title"       -> текст обрезается до 512, инференс 1 раз
        mode="publication" -> режем на чанки и агрегируем
        """
        if mode == "title":
            res = self.pipe(text, truncation=True, max_length=self.max_length)
            if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                res = res[0]
            out = self._postprocess_single(res)
        else:
            chunks = self._chunk_text(text)
            preds = []
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i+self.batch_size]
                res_batch = self.pipe(batch, truncation=True, max_length=self.max_length)
                for scores in res_batch:
                    preds.append(self._postprocess_single(scores))
            out = self._aggregate(preds)

        if meta is not None:
            out["meta"] = meta
        return out

    def predict_batch(self,
                      texts: List[str],
                      mode: str = "title",
                      metas: Optional[List[dict]] = None) -> List[dict]:
        """
        Батч инференс.
        - mode="title"       -> обычный батч
        - mode="publication" -> каждый текст режется и агрегируется
        """
        results = []
        if mode == "title":
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                res_batch = self.pipe(batch, truncation=True, max_length=self.max_length)
                for scores in res_batch:
                    results.append(self._postprocess_single(scores))
        else:
            for text in texts:
                results.append(self.predict(text, mode="publication"))

        if metas is not None:
            for r, m in zip(results, metas):
                r["meta"] = m

        return results
    

class EmotionDetector:
    """
    Обёртка для модели эмоций:
      - 'Djacon/rubert-tiny2-russian-emotion-detection' по умолчанию.
    Методы:
      - predict(text, mode="title"|"publication")
      - predict_batch(texts, mode=...)
    Добавляет 'volatility_score' = суммарная вероятность меток из self.volatility_labels.
    """

    def __init__(self,
                 model_name: str = "Djacon/rubert-tiny2-russian-emotion-detection",
                 device: Optional[int] = None,
                 batch_size: int = 16,
                 max_length: int = 512,
                 volatility_labels: Optional[List[str]] = None):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=self.tokenizer,
            device=device if device >= 0 else -1,
            return_all_scores=True
        )
        self.volatility_labels = volatility_labels or ["fear", "surprise", "angry", "sad"]

    def _postprocess_single(self, scores):
        out = {d["label"]: float(d["score"]) for d in scores}
        best_label = max(out.items(), key=lambda x: x[1])[0]
        vol = sum(out.get(lab, 0.0) for lab in self.volatility_labels)
        return {
            "label": best_label,
            "scores": out,
            "confidence": out[best_label],
            "volatility_score": vol
        }

    def _chunk_text(self, text: str, stride: int = 256) -> List[str]:
        """Режет длинный текст на чанки длиной <= max_length с перекрытием."""
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        for i in range(0, len(tokens), stride):
            window = tokens[i:i + self.max_length]
            if not window:
                continue
            chunks.append(self.tokenizer.convert_tokens_to_string(window))
            if i + self.max_length >= len(tokens):
                break
        return chunks

    def _aggregate(self, preds: List[dict]) -> dict:
        """Агрегирует предсказания по чанкам (усредняем score и volatility)."""
        if not preds:
            return {}
        labels = preds[0]["scores"].keys()
        agg_scores = {lab: 0.0 for lab in labels}
        vol_total = 0.0
        for p in preds:
            for lab, val in p["scores"].items():
                agg_scores[lab] += val
            vol_total += p["volatility_score"]
        for lab in agg_scores:
            agg_scores[lab] /= len(preds)
        vol_total /= len(preds)
        best_label = max(agg_scores.items(), key=lambda x: x[1])[0]
        return {"label": best_label, "scores": agg_scores,
                "confidence": agg_scores[best_label],
                "volatility_score": vol_total}

    def predict(self, text: str, mode: str = "title", meta: Optional[dict] = None) -> dict:
        if mode == "title":
            res = self.pipe(text, truncation=True, max_length=self.max_length)
            if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                res = res[0]
            out = self._postprocess_single(res)
        else:
            chunks = self._chunk_text(text)
            preds = []
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i+self.batch_size]
                res_batch = self.pipe(batch, truncation=True, max_length=self.max_length)
                for scores in res_batch:
                    preds.append(self._postprocess_single(scores))
            out = self._aggregate(preds)

        if meta is not None:
            out["meta"] = meta
        return out

    def predict_batch(self,
                      texts: List[str],
                      mode: str = "title",
                      metas: Optional[List[dict]] = None) -> List[dict]:
        results = []
        if mode == "title":
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                res_batch = self.pipe(batch, truncation=True, max_length=self.max_length)
                for scores in res_batch:
                    results.append(self._postprocess_single(scores))
        else:
            for text in texts:
                results.append(self.predict(text, mode="publication"))

        if metas is not None:
            for r, m in zip(results, metas):
                r["meta"] = m
        return results