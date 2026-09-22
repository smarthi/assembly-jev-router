# Assembly Jev Router

[![Language: Assembly](https://img.shields.io/badge/language-Assembly-blue)](https://github.com/smarthi/assembly-jev-router)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon%20%28ARM64%29-blue)](#)

A high-performance, deterministic classifier and router implementation in Apple Silicon (AArch64) assembly language, inspired by TypeSafe AI's Jev typed-decision interface.

## Overview

This repository implements a lightweight decision engine that maps integer feature vectors to predefined classes with confidence scoring and optional routing through type-safe handlers. The implementation is:

- **Deterministic**: Uses only signed integer arithmetic for classification (no floating-point approximation)
- **Allocation-free**: Requires no runtime memory allocation
- **Type-safe**: Enforces structured input/output through C-compatible structs
- **Fast**: Optimized for ARM64/AArch64 architecture on Apple Silicon (macOS)

This is **not** an implementation of TypeSafe AI's proprietary Jev model, but rather a standalone decision primitive inspired by its typed interface.

## Features

### Core Functionality

- **Classification**: Scores a feature vector against trained weight matrices and biases
- **Confidence Scoring**: Computes a confidence heuristic (in basis points: 0-10000) based on score margins
- **Routing**: Dispatches to class-indexed handler functions with typed contexts
- **Result Reporting**: Provides the winning class, best and runner-up scores

### Technical Highlights

- Pure ARM64 assembly with zero external dependencies
- Follows Apple's AArch64 calling convention (AAPCS64)
- Handles edge cases: null pointer validation, single-class models, missing handlers
- Floating-point used only for confidence normalization (classification is pure integer)

## Data Structures

### `jev_model`

Represents a trained classification model:

```c
typedef struct {
    uint64_t class_count;           // Number of classes
    uint64_t feature_count;         // Number of features per sample
    const int32_t *weights;         // Class-major weight matrix (C × F)
    const int64_t *biases;          // Class biases (C elements)
    const void *const *handlers;    // Optional function pointers (C elements)
} jev_model;
