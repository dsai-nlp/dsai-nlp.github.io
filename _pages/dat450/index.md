---
layout: page
title: DAT450/DIT247 Assignments
permalink: /courses/dat450/
description:
nav: false
---

<style>
  .dat450-hero {
    background: linear-gradient(135deg, rgba(10, 95, 166, 0.08), rgba(95, 28, 160, 0.08));
    border-radius: 2rem;
    padding: 2.8rem 2.4rem;
    margin-bottom: 2.6rem;
  }

  .dat450-hero h1 {
    font-size: clamp(2.1rem, 4.5vw, 2.8rem);
    font-weight: 700;
    margin-bottom: 0.8rem;
  }

  .dat450-hero p {
    font-size: 1rem;
    color: var(--text-muted);
    max-width: 720px;
    margin-bottom: 0;
  }

  .dat450-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.8rem;
  }

  .dat450-card {
    background-color: var(--global-card-bg-color, #fff);
    border-radius: 1.3rem;
    padding: 1.9rem 1.6rem;
    box-shadow: 0 16px 32px rgba(0, 0, 0, 0.08);
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    font-size: 0.95rem;
    line-height: 1.6;
  }

  .dat450-card h2 {
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
  }

  .dat450-card a {
    color: inherit;
    text-decoration: underline;
  }

  @media (max-width: 576px) {
    .dat450-hero {
      padding: 2.3rem 1.8rem;
    }

    .dat450-card {
      padding: 1.6rem 1.4rem;
    }
  }
</style>

<section class="dat450-hero">
  <h1>DAT450/DIT247 coursework</h1>
  <p>Programming assignments and the project brief for Machine Learning for Natural Language Processing.</p>
</section>

<div class="dat450-grid">
  <article class="dat450-card">
    <h2>Programming Assignment 1</h2>
    <p>Introduction to language modeling: build a neural language model, handle preprocessing, and explore embeddings.</p>
    <a href="{{ '/courses/dat450/assignment1/' | relative_url }}">Open assignment 1</a>
  </article>
  <article class="dat450-card">
    <h2>Programming Assignment 2</h2>
    <p>Sequence labelling with neural networks, focusing on tagging architectures and training routines.</p>
    <a href="{{ '/courses/dat450/assignment2/' | relative_url }}">Open assignment 2</a>
  </article>
  <article class="dat450-card">
    <h2>Programming Assignment 4</h2>
    <p>Neural machine translation with encoder-decoder models and attention mechanisms.</p>
    <a href="{{ '/courses/dat450/assignment4/' | relative_url }}">Open assignment 4</a>
  </article>
  <article class="dat450-card">
    <h2>Programming Assignment 5</h2>
    <p>Abstractive summarisation using encoder-decoder architectures and evaluation techniques.</p>
    <a href="{{ '/courses/dat450/assignment5/' | relative_url }}">Open assignment 5</a>
  </article>
  <article class="dat450-card">
    <h2>Course project</h2>
    <p>Guidelines for the independent project, including deliverables, milestones, and reporting format.</p>
    <a href="{{ '/courses/dat450/project/' | relative_url }}">View project brief</a>
  </article>
</div>
