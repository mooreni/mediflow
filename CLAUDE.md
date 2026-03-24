# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MediFlow** is a medical document translation engine POC designed to help immigrants in Israel understand medical documents in their native language. It is a multi-agent AI system built with DSPy and Google Gemini models.

This project is currently in the **planning phase**. The requirements specification lives in `planning/requirements.md`.

## Architecture

The system is a pipeline with four major components:

### 1. Evaluation Mechanism (Milestone 1)
- **Comet evaluator**: HuggingFace Comet model for semantic similarity scoring against a gold-standard translation
- **LLM-as-a-Judge**: Gemini Pro with a medical-specific rubric (1–10 scale, emphasis on medical terms, dosages, diagnoses)
- **Hybrid approach**: Both evaluators run in parallel; correlation is analyzed to pick the DSPy optimization metric

### 2. Agent Orchestration & Classification (Milestone 3)
- **Router Agent**: Lightweight Gemini Flash agent that classifies input documents into one of 5 types: `Summary`, `Prescript`, `Referral`, `Form`, `Record`
- **DSPy Signatures**: 5 separate signatures with type-specific translation instructions (e.g., drug names left untranslated in prescriptions)
- **Optimizer**: DSPy optimizer runs per-agent using the evaluation mechanism to tune prompts

### 3. Transparency / XAI (Milestone 4)
- **Confidence extraction**: Logprobs per token, or a reflection function where the model self-reviews and returns uncertain words
- **Tagging**: Uncertain words are marked (special Markdown tag or similar) so the UI can highlight them

### 4. OCR & UI Integration (Milestone 5)
- **OCR**: Google Cloud Vision API (`TEXT_DETECTION`) for Hebrew text extraction from images/scans
- **Pipeline**: `Image → OCR → Router → Specialist Agent → Tagged Translation`
- **Frontend**: File upload UI showing loading state and final translated document with highlighted uncertain words

## Tech Stack
- **LLM Framework**: DSPy
- **Models**: Google Gemini Pro (evaluation/translation), Gemini Flash (routing)
- **Evaluation**: HuggingFace Comet
- **OCR**: Google Cloud Vision API
- **Languages**: Medical documents are primarily Hebrew; output is multilingual
