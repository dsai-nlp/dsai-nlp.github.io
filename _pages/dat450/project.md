---
layout: page
title: 'DAT450/DIT247: Independent project'
permalink: /courses/dat450/project/
description:
nav: false
nav_order: 4
---

# DAT450/DIT247: Independent project

## Your work

In the project work, your task will be to

- find an interesting task that can be addressed with machine learning methods
- find (or annotate) an appropriate dataset for that task
- look around for related work
- implement software to train some machine learning model for your task
- evaluate your system
- write up a report about your results.

The projects are done in groups of up to four.

## Preliminary project description

Please submit a short text that describes what you intend to work on. For suggestions of topics, see below.

You should submit this description [in Canvas](https://chalmers.instructure.com/courses/31739/assignments/98459). The text should include a project title and one or a couple of paragraphs giving a rough outline of how you intend to work. If you have found any related work or relevant dataset, this is useful information to include. The deadline for this preliminary description is **December 6**.

It is not compulsory to submit the preliminary description, but we strongly recommend it so that we have the opportunity to give you some initial feedback. 

## The report and the presentation

When writing your report, please use one of these style templates (available for Word and LaTeX). The text should be structured as a typical academic research paper, including an abstract, statement of problem, method description, experiments and results, and a conclusion. We will not enforce any page limits strictly, but we think that the optimal length of a report is something similar to a typical short research paper: around 4 pages in a two-column format, plus references.

For examples of how such a text can look, take a look at some [short papers published at the ACL conference](https://aclanthology.org/events/acl-2024/#2024acl-short). In addition, there are some examples of project reports from previous instances of this course provided [in this directory](https://www.cse.chalmers.se/~richajo/dat450/assignments/examples/). (Please note that the 2020 examples used another document style.)

Please submit the project report via the Canvas page. The deadline for submitting the report is January XX.

In addition, you should prepare a presentation of up to 10 minutes. You should create a recording of this presentation and make it available for other students to see. You can present the project even if your project work and report aren't finished. Please submit the recording of your oral presentation via this Canvas page. The deadline for making your recording available is January YY. Finally, you will be expected to watch the presentations by at least two other groups and provide some feedback. 

## Grading

The grading is primarily based on the quality of the written report, and to a smaller degree on the quality of the presentation. For a maximal score, the writing is expected to be concise and clear, related work to be described thoroughly, the technical approach to be evaluated systematically, and the conclusions to be insightful.

It is not necessarily the case that a more "difficult" project will give a high grade, but maybe it will be easier to come up with interesting questions for experiments and analysis if the topic isn't too similar to the course assignments or lecture demos. 

## Finding a topic

We encourage people to find a topic that they have a personal interest in, if possible. You can work on any project that is small and manageable enough for a short project of this kind. It's OK if you end up with preliminary results, as long as you have some tidbit to present at the end.

If you need help deciding on a topic, here are a few suggestions. Some of these are very easy and some more challenging, and this list is not sorted by the difficulty level.

- Adapting a lab assignment or lecture demo. Start from one of the lab assignments or lecture demos and change it to something that is interesting for you. Maybe change the model in Assignment 2 to a Transformer? Or compare a neural topic model to the Gibbs sampling algorithm in Assignment 3?
- Comparing representations for an application. Select some NLP application, either one we saw during the lectures or something where you can find the code online. Investigate how various types of text representations affect the application's performance. For instance, you could compare standard word embeddings, ELMo, BERT, and other representations.
- Machine translation for a language pair of your choice.
- Retreival-augmented generation in an application domain of your choice. Maybe you'd like a chatbot that can answer questions about the courses available at the university?
- Caption generation. Build a system that generates descriptions of images.
- Entity and/or relation extraction using a dataset such as [DrugProt](https://zenodo.org/record/5042151) or something else.
- Knowledge distillation. Modern representation models for language are huge and slow. Can you use knowledge distillation or some other compression technique to develop something that is more efficient (in some application)?
- Shared tasks. Every year, several NLP competitions are arranged at the SemEval conference. Participate in one of the 2025 tasks or get the data from some task from previous years. Additionally, there are shared tasks organized by CoNLL, BioNLP and other conferences, where datasets can often be accessed easily. 

Alternatively, if you don't have an idea and you don't like the suggested topics, just talk to the teachers in the course. 

## Frequently asked questions

**Q:** Can I use someone else's software in my project.

**A:** Generally, yes. In particular, it is completely standard to use various libraries (e.g. HuggingFace) and pre-trained models. But your project work can't be *only* that software: your addition has to be nontrivial. For instance, if you work on image captioning, you can't just take an existing image captioning implementation and submit it as your own work.

**Q:** Are we allowed to use commercial large language models (such as one of the OpenAI models)?

**A:** Yes, but there are a few things to note here. 1) You have to pay for it out of your own pocket. 2) The way you apply it needs to be "nontrivial" (let's discuss if you are unsure). 3) Evaluations need to be as rigorous as in any project. 4) You should save your outputs so that your evaluation scores are reproducible. 

