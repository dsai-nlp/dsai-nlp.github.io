---
layout: members
title: Members
permalink: /members
nav: true
nav_order: 1
---

<div class="members-page">
<style>
  .members-page .members-hero {
    background: linear-gradient(135deg, rgba(10, 95, 166, 0.08), rgba(95, 28, 160, 0.08));
    border-radius: 2rem;
    padding: 3rem 2.5rem;
    margin-bottom: 3rem;
  }

  .members-page .members-hero h1 {
    font-size: clamp(2.2rem, 5vw, 3rem);
    font-weight: 700;
    margin-bottom: 0.75rem;
  }

  .members-page .members-hero p {
    font-size: 1.05rem;
    color: var(--text-muted);
    max-width: 720px;
    margin-bottom: 0;
  }

  .members-page .members-section {
    background-color: rgba(0, 0, 0, 0.02);
    border-radius: 1.5rem;
    padding: 2.5rem 2rem;
    margin-bottom: 3rem;
  }

  .members-page .members-section:last-of-type {
    margin-bottom: 0;
  }

  .members-page .members-section .section-title {
    text-align: center;
    margin-bottom: 2.5rem;
  }

  .members-page .members-grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 2.5rem;
  }

  .members-page .member-card {
    width: 220px;
    text-align: center;
    position: relative;
  }

  .members-page .member-card a {
    color: inherit;
    text-decoration: none;
  }

  .members-page .member-avatar {
    width: 160px;
    height: 160px;
    border-radius: 50%;
    object-fit: cover;
    box-shadow: 0 16px 32px rgba(0, 0, 0, 0.12);
    border: 6px solid rgba(255, 255, 255, 0.75);
    margin: 0 auto 1.25rem auto;
    display: block;
    background-color: rgba(0, 0, 0, 0.05);
  }

  .members-page .member-name {
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 0.35rem;
  }

  .members-page .member-role {
    color: var(--text-muted);
    margin-bottom: 0.35rem;
  }

  .members-page .member-contact {
    color: var(--text-muted);
    font-size: 0.9rem;
    word-break: break-word;
  }

  @media (max-width: 576px) {
    .members-page .member-card {
      width: 220px;
    }

    .members-page .member-avatar {
      width: 130px;
      height: 130px;
    }
  }
</style>

<section class="members-hero">
  <h1>Meet the NLP@DSAI Team</h1>
  <p>Explore the people driving our research — from faculty and researchers to the students and alumni who keep our community vibrant.</p>
</section>

{% assign current_members = site.members | where: "state", "current" | sort: "name" %}
{% assign staff_members = "" | split: "" %}
{% assign student_members = "" | split: "" %}
{% assign other_members = "" | split: "" %}
{% for member in current_members %}
  {% assign position = member.position | downcase %}
  {% if position contains "professor" or position contains "lecturer" or position contains "director" %}
    {% assign staff_members = staff_members | push: member %}
  {% elsif position contains "phd" or position contains "student" or position contains "intern" %}
    {% assign student_members = student_members | push: member %}
  {% else %}
    {% assign other_members = other_members | push: member %}
  {% endif %}
{% endfor %}

{% if staff_members.size > 0 %}
<section class="members-section">
  <div class="section-title">
    <h2 class="h3 mb-1">Staff</h2>
    <p class="text-muted mb-0">Faculty and senior researchers leading NLP@DSAI</p>
  </div>
  <div class="members-grid">
    {% for member in staff_members %}
    {% assign avatar = member.image | default: 'assets/members/placeholder.png' %}
    {% if avatar contains 'http://' or avatar contains 'https://' %}
      {% assign avatar_url = avatar %}
    {% else %}
      {% assign avatar_url = avatar | relative_url %}
    {% endif %}
    <div class="member-card">
      <a href="{{ member.url | relative_url }}">
        <img class="member-avatar" src="{{ avatar_url }}" alt="{{ member.name }}">
        <div class="member-name">{{ member.name }}</div>
        {% if member.position %}
        <div class="member-role">{{ member.position }}</div>
        {% endif %}
        {% if member.email %}
        <div class="member-contact">{{ member.email }}</div>
        {% endif %}
      </a>
    </div>
    {% endfor %}
  </div>
</section>
{% endif %}

{% if student_members.size > 0 %}
<section class="members-section">
  <div class="section-title">
    <h2 class="h3 mb-1">PhD Researchers</h2>
    <p class="text-muted mb-0">Doctoral candidates advancing cutting-edge NLP research</p>
  </div>
  <div class="members-grid">
    {% for member in student_members %}
    {% assign avatar = member.image | default: 'assets/members/placeholder.png' %}
    {% if avatar contains 'http://' or avatar contains 'https://' %}
      {% assign avatar_url = avatar %}
    {% else %}
      {% assign avatar_url = avatar | relative_url %}
    {% endif %}
    <div class="member-card">
      <a href="{{ member.url | relative_url }}">
        <img class="member-avatar" src="{{ avatar_url }}" alt="{{ member.name }}">
        <div class="member-name">{{ member.name }}</div>
        {% if member.position %}
        <div class="member-role">{{ member.position }}</div>
        {% endif %}
        {% if member.email %}
        <div class="member-contact">{{ member.email }}</div>
        {% endif %}
      </a>
    </div>
    {% endfor %}
  </div>
</section>
{% endif %}

{% if other_members.size > 0 %}
<section class="members-section">
  <div class="section-title">
    <h2 class="h3 mb-1">Research Team</h2>
    <p class="text-muted mb-0">Collaborators and associates supporting our initiatives</p>
  </div>
  <div class="members-grid">
    {% for member in other_members %}
    {% assign avatar = member.image | default: 'assets/members/placeholder.png' %}
    {% if avatar contains 'http://' or avatar contains 'https://' %}
      {% assign avatar_url = avatar %}
    {% else %}
      {% assign avatar_url = avatar | relative_url %}
    {% endif %}
    <div class="member-card">
      <a href="{{ member.url | relative_url }}">
        <img class="member-avatar" src="{{ avatar_url }}" alt="{{ member.name }}">
        <div class="member-name">{{ member.name }}</div>
        {% if member.position %}
        <div class="member-role">{{ member.position }}</div>
        {% endif %}
        {% if member.email %}
        <div class="member-contact">{{ member.email }}</div>
        {% endif %}
      </a>
    </div>
    {% endfor %}
  </div>
</section>
{% endif %}

{% assign master_members = site.members | where: "state", "master" | sort: "name" %}
{% if master_members.size > 0 %}
<section class="members-section">
  <div class="section-title">
    <h2 class="h3 mb-1">Master&apos;s Students</h2>
    <p class="text-muted mb-0">Emerging researchers contributing through their thesis work</p>
  </div>
  <div class="members-grid">
    {% for member in master_members %}
    {% assign avatar = member.image | default: 'assets/members/placeholder.png' %}
    {% if avatar contains 'http://' or avatar contains 'https://' %}
      {% assign avatar_url = avatar %}
    {% else %}
      {% assign avatar_url = avatar | relative_url %}
    {% endif %}
    <div class="member-card">
      <a href="{{ member.url | relative_url }}">
        <img class="member-avatar" src="{{ avatar_url }}" alt="{{ member.name }}">
        <div class="member-name">{{ member.name }}</div>
        {% if member.position %}
        <div class="member-role">{{ member.position }}</div>
        {% endif %}
        {% if member.email %}
        <div class="member-contact">{{ member.email }}</div>
        {% endif %}
      </a>
    </div>
    {% endfor %}
  </div>
</section>
{% endif %}

{% assign alumni_members = site.members | where: "state", "alumni" | sort: "name" %}
{% if alumni_members.size > 0 %}
<section class="members-section">
  <div class="section-title">
    <h2 class="h3 mb-1">Alumni</h2>
    <p class="text-muted mb-0">Former members who continue to shape the broader NLP community</p>
  </div>
  <div class="members-grid">
    {% for member in alumni_members %}
    {% assign avatar = member.image | default: 'assets/members/placeholder.png' %}
    {% if avatar contains 'http://' or avatar contains 'https://' %}
      {% assign avatar_url = avatar %}
    {% else %}
      {% assign avatar_url = avatar | relative_url %}
    {% endif %}
    <div class="member-card">
      <a href="{{ member.url | relative_url }}">
        <img class="member-avatar" src="{{ avatar_url }}" alt="{{ member.name }}">
        <div class="member-name">{{ member.name }}</div>
        {% if member.position %}
        <div class="member-role">{{ member.position }}</div>
        {% endif %}
        {% if member.email %}
        <div class="member-contact">{{ member.email }}</div>
        {% endif %}
      </a>
    </div>
    {% endfor %}
  </div>
</section>
{% endif %}
</div>
