---
layout: page
title: Courses
permalink: /courses/
description:
nav: true
nav_order: 5
---

<style>
  .courses-hero {
    background: linear-gradient(135deg, rgba(10, 95, 166, 0.08), rgba(95, 28, 160, 0.08));
    border-radius: 2rem;
    padding: 3rem 2.5rem;
    margin-bottom: 3rem;
  }

  .courses-hero h1 {
    font-size: clamp(2.2rem, 5vw, 3rem);
    font-weight: 700;
    margin-bottom: 0.9rem;
  }

  .courses-hero p {
    font-size: 1.05rem;
    color: var(--text-muted);
    max-width: 760px;
    margin-bottom: 0;
  }

  .courses-section {
    background-color: rgba(0, 0, 0, 0.02);
    border-radius: 1.5rem;
    padding: 2.5rem 2rem;
    margin-bottom: 3rem;
  }

  .courses-section:last-of-type {
    margin-bottom: 0;
  }

  .courses-section .section-title {
    text-align: center;
    margin-bottom: 2.3rem;
  }

  .courses-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 2rem;
  }

  .course-card {
    background-color: var(--global-card-bg-color, #fff);
    border-radius: 1.3rem;
    padding: 2rem 1.75rem;
    box-shadow: 0 18px 36px rgba(0, 0, 0, 0.08);
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    font-size: 0.95rem;
    line-height: 1.65;
  }

  .course-card h3 {
    font-size: 1.2rem;
    font-weight: 600;
    margin: 0;
  }

  .course-card p {
    margin: 0;
  }

  .course-card a {
    color: inherit;
    text-decoration: underline;
  }

  .course-links-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 2rem;
  }

  .course-links-card {
    background-color: var(--global-card-bg-color, #fff);
    border-radius: 1.3rem;
    padding: 2rem 1.75rem;
    box-shadow: 0 16px 32px rgba(0, 0, 0, 0.08);
    display: flex;
    flex-direction: column;
    gap: 1rem;
    font-size: 0.95rem;
    line-height: 1.6;
  }

  .course-links-card h3 {
    font-size: 1.15rem;
    font-weight: 600;
    margin: 0;
  }

  .course-links-card a {
    color: inherit;
    text-decoration: underline;
  }

  @media (max-width: 576px) {
    .courses-hero {
      padding: 2.4rem 1.8rem;
    }

    .course-card {
      padding: 1.8rem 1.5rem;
    }
  }
</style>

<section class="courses-hero">
  <h1>Courses in Natural Language Processing</h1>
  <p>Our courses in NLP cover the most common technical architectures used in modern NLP, and describe technical solutions for NLP applications such as categorization, entity recognition, translation, and summarization. We have a hands-on approach in the coursework where we implement some classical algorithms as well as the most recent LLM-based techniques.</p>
</section>

<section class="courses-section">
  <div class="section-title">
    <h2 class="h3 mb-1">Current courses</h2>
  </div>
  <div class="courses-grid">
    <article class="course-card">
      <h3>Machine Learning for Natural Language Processing</h3>
      <p><strong>Machine Learning for Natural Language Processing</strong>. This Master-level course is offered to students at <a href="https://www.chalmers.se/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/DAT450/?acYear=2023/2024" target="_blank" rel="noopener">Chalmers University of Technology</a> and the <a href="https://www.gu.se/studera/hitta-utbildning/maskininlarning-for-sprakteknologi-dit247" target="_blank" rel="noopener">University of Gothenburg</a> and is open to students with a previous introduction to machine learning. It includes four compulsory assignments and an independent project.</p>
    </article>
    <article class="course-card">
      <h3>Deep Learning for Natural Language Processing</h3>
      <p><a href="https://liu-nlp.github.io/dl4nlp/" target="_blank" rel="noopener"><strong>Deep Learning for Natural Language Processing</strong></a>. This course is open to PhD students within the WASP graduate school who have some previous background in machine learning. It is taught jointly with Marco Kuhlmann at Linköping University. It includes three compulsory assignments and an independent project.</p>
    </article>
  </div>
</section>

<section class="courses-section">
  <div class="section-title">
    <h2 class="h3 mb-1">Master's thesis supervision</h2>
  </div>
  <div class="courses-grid">
    <article class="course-card">
      <p>If you are a GU or Chalmers student who wants to carry out a Master's Thesis project on a natural language processing topic, please get in touch and we can meet for a discussion. We are quite open to supervising projects in the general NLP area.</p>
      <p>In addition, please take a look at <a href="https://chalmers.instructure.com/courses/232" target="_blank" rel="noopener">the CSE department's guidelines for thesis projects</a> so that you are aware of the relevant regulations and deadlines. Please note that the department strongly prefers thesis projects to be carried out in pairs.</p>
      <p>Keep in mind that it's much more likely to find a willing supervisor if you have a research-oriented topic, although we can also take on thesis topics defined by companies, as long as we can find a relevant research angle.</p>
    </article>
  </div>
</section>

<section class="courses-section">
  <div class="section-title">
    <h2 class="h3 mb-1">Course resources</h2>
  </div>
  <div class="course-links-grid">
    <article class="course-links-card">
      <h3>Course DAT450/DIT247</h3>
      <p>Browse the programming assignments, course project, and supplementary instructions for Machine Learning for Natural Language Processing.</p>
      <a href="{{ '/courses/dat450/' | relative_url }}">View DAT450/DIT247 materials</a>
    </article>
    <article class="course-links-card">
      <h3>Course Deep Learning for NLP</h3>
      <p>PhD course material, schedule, and assignment descriptions are hosted on the joint course website maintained together with Linköping University.</p>
      <a href="https://liu-nlp.github.io/dl4nlp/" target="_blank" rel="noopener">Open DL4NLP site</a>
    </article>
  </div>
</section>
