---
title: "Sweep: pretraining ablations"
date: 2026-11-22
categories: []
tags: []
description: "Pretraining ablations on femtochat: what each optimizer, initialization, and architecture choice is worth."
---

> Draft scaffold. The prose below is a placeholder — this file exists so the
> figure path (`make_figures.py` → `femtochat/<post>/output/figures/` →
> `assets/img/femtochat/<post>/` → this post) is proven before the real post is
> written. The numbers live in
> `femtochat/xxxx-xx-xx-pretraining-ablations-of-femtochat/REPORT.md`.

## Introduction

Each run changes one setting against a matched control, and the plots are
regenerated from the training event files by `make_figures.py`.

## Optimizer and initialization

![Optimizer, initialization & weight tying](/assets/img/femtochat/xxxx-xx-xx-pretraining-ablations-of-femtochat/sweep_optimizer_init.png)

## Closing

Placeholder.
