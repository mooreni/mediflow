# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MediFlow** is a medical document translation engine POC designed to help immigrants in Israel understand medical documents in their native language. It is a multi-agent AI system built with DSPy and Google Gemini models.

This project is currently in the **planning phase**. The requirements specification lives in `planning/requirements.md`.

## Architecture

The system is a pipeline with four major components:

### 1. Evaluation Mechanism (Milestone 1)
- **LLM-as-a-Judge**: Gemini Pro 3.1 with a medical-specific rubric (0–100 scale, emphasis on medical terms, dosages, diagnoses); this is the sole evaluator and the DSPy optimization metric

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
- **Evaluation**: Google Gemini Pro
- **OCR**: TBD
- **Languages**: Medical documents are primarily Hebrew; output is mainly russian.

## Rules
- IMPORTANT: Before asking to perform an action (like reading a file or changing code), explain briefly what is it needed for.
See [RULES.md](./RULES.md) for all coding standards, design principles, error handling, testing, security, and self-review rules.
