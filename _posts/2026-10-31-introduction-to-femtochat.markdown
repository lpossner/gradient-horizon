---
layout: post
title:  "Introduction to Femtochat"
date:   2026-10-31
categories: jekyll update
---
This post introduces my side project, `Femtochat`, which aims to be a simpler version of [Nanochat][karpathy-github]. It is a personal learning project that I want to share with the world. I want to understand how LLMs work under the hood and what the training process looks like. Sharing the code and writing about it creates some external pressure to actually work on it. I hope I can stop procrastinating, publish some results, and, most importantly, learn a lot. Right now, everything is still a bit of a mess, so let's see where the project goes.

I was trained as an electrical engineer and got into machine learning fairly late, in January 2023—just a few months after ChatGPT was released. I approached the topic by understanding and implementing basic algorithms on my own: linear regression, decision trees, SVMs, and feed-forward and convolutional neural networks. Like many of you, I was, and still am, fascinated (and occasionally unsettled) by the capabilities that LLMs and other transformer-based models have demonstrated in recent years.

To learn more about LLMs, I worked through Andrej Karpathy's [videos][karpathy-channel] and his project [Nanochat][karpathy-github], as well as Sebastian Raschka's [books][raschka-page] and his [Substack][raschka-substack]. All of them are excellent resources.

[Nanochat][karpathy-github] is an end-to-end implementation of a transformer-based LLM with a decoder architecture, which serves as the foundation for almost all frontier models.

Rather than just copying it, I wanted to create something from scratch, but in a more educational flavor and with lower complexity. Hence, I set out to build a humbler version named `Femtochat`.

In the following posts, I will blog about `Femtochat` and show my progress.

[karpathy-channel]: https://www.youtube.com/@AndrejKarpathy/
[karpathy-github]: https://github.com/karpathy/nanochat/
[raschka-page]: https://www.manning.com/authors/sebastian-raschka
[raschka-substack]: https://magazine.sebastianraschka.com
