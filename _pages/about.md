---
layout: default
title: NLP@DSAI
permalink: /
subtitle:

news: false
latest_posts: false
selected_papers: false
social: false
---

<style>
  .home-section {
    color: inherit;
  }

  .hero-section {
    background: linear-gradient(135deg, rgba(10, 95, 166, 0.08), rgba(95, 28, 160, 0.08));
    border-radius: 2rem;
    padding: 3rem 2.5rem;
  }

  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    background-color: rgba(255, 255, 255, 0.6);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    color: #0a5fa6;
    margin-bottom: 1.5rem;
  }

  .hero-title {
    font-size: clamp(2.2rem, 5vw, 3rem);
    margin-bottom: 1rem;
    font-weight: 700;
  }

  .hero-text {
    font-size: 1.1rem;
    line-height: 1.7;
    color: var(--text-muted);
  }

  .hero-logos {
    display: flex;
    gap: 1.5rem;
    justify-content: center;
    align-items: center;
    flex-wrap: wrap;
    max-width: 360px;
    margin: 2rem auto 0;
  }

  .hero-logo-card {
    background-color: rgba(255, 255, 255, 0.78);
    border-radius: 1.25rem;
    padding: 1rem;
    width: 150px;
    aspect-ratio: 1;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    box-shadow: 0 16px 32px rgba(0, 0, 0, 0.08);
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .hero-logo-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 36px rgba(0, 0, 0, 0.12);
  }

  .hero-logo-card img {
    max-width: 85%;
    max-height: 75%;
    object-fit: contain;
    filter: saturate(0.1);
  }

  .funding-section {
    background-color: rgba(0, 0, 0, 0.02);
    border-radius: 1.5rem;
    padding: 2.5rem 2rem;
  }
  .funding-section .section-title {
    text-align: center;
    margin-bottom: 2.5rem;
  }

  .funding-logos {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 2rem;
    justify-items: center;
  }

  .funding-logo-card {
    width: 100%;
    aspect-ratio: 1;
    text-align: center;
    padding: 1.5rem;
    border-radius: 1.25rem;
    background-color: #fff;
    box-shadow: 0 16px 32px rgba(0, 0, 0, 0.08);
    transition: transform 0.2s ease;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  .funding-logo-card:hover {
    transform: translateY(-4px);
  }

  .funding-logo-card img {
    max-width: 75%;
    max-height: 60%;
    object-fit: contain;
    margin-bottom: 1rem;
    filter: saturate(0.15);
  }

  .funding-caption {
    font-size: 0.95rem;
    color: var(--text-muted);
    padding: 0 0.5rem;
  }
  

  .news-section {
    background-color: rgba(0, 0, 0, 0.02);
    border-radius: 1.5rem;
    padding: 2.5rem 2rem;
  }

  .news-section .section-title {
    text-align: center;
    margin-bottom: 2.5rem;
  }

  .news-grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 2.5rem;
  }

  .news-card {
    width: 280px;
    text-align: center;
    padding: 2rem 1.5rem;
    border-radius: 1.4rem;
    background-color: var(--global-card-bg-color, #fff);
    box-shadow: 0 16px 32px rgba(0, 0, 0, 0.08);
    transition: transform 0.2s ease;
  }

  .news-card:hover {
    transform: translateY(-6px);
  }

  .news-card a {
    color: inherit;
    text-decoration: none;
  }

  .news-card h3 {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  .news-date {
    color: var(--text-muted);
    margin-bottom: 0.5rem;
  }

  .news-excerpt {
    color: var(--text-muted);
    font-size: 0.95rem;
  }

  .discussions-section {
    background-color: rgba(0, 0, 0, 0.02);
    border-radius: 1.5rem;
    padding: 2.5rem 2rem;
  }

  .discussions-section .section-title {
    text-align: center;
    margin-bottom: 2.2rem;
  }

  .discussions-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.75rem;
  }

  .discussion-card {
    display: block;
    padding: 1.75rem 1.5rem;
    border-radius: 1.4rem;
    background-color: var(--global-card-bg-color, #fff);
    box-shadow: 0 16px 32px rgba(0, 0, 0, 0.08);
    color: inherit;
    text-decoration: none;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }

  .discussion-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
  }

  .discussion-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 0.45rem;
  }

  .discussion-meta {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-bottom: 0.65rem;
  }

  .discussion-preview {
    font-size: 0.95rem;
    color: var(--text-muted);
  }

  .discussion-details {
    list-style: none;
    padding: 0;
    margin: 1.2rem 0 0;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .discussion-detail {
    font-size: 0.95rem;
    color: var(--global-text-color);
    line-height: 1.5;
  }

  .discussion-detail strong {
    font-weight: 600;
    margin-right: 0.35rem;
  }

  .campus-section {
    background-color: rgba(0, 0, 0, 0.02);
    border-radius: 1.5rem;
    padding: 2.5rem 2rem;
  }

  .campus-section .section-title {
    text-align: center;
    margin-bottom: 2.5rem;
  }

  .campus-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
    align-items: stretch;
  }

  .campus-card {
    background-color: var(--global-card-bg-color, #fff);
    border-radius: 1.3rem;
    padding: 2rem 1.6rem;
    box-shadow: 0 16px 32px rgba(0, 0, 0, 0.08);
  }

  .campus-card h3 {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.85rem;
  }

  .campus-card ul {
    list-style: none;
    padding: 0;
    margin: 0;
    display: grid;
    gap: 0.75rem;
    font-size: 0.95rem;
  }

  .campus-card li span {
    display: block;
    color: var(--text-muted);
    font-size: 0.88rem;
    margin-top: 0.15rem;
  }

  .campus-map-frame {
    position: relative;
    padding-bottom: 60%;
    height: 0;
    overflow: hidden;
    border-radius: 1.3rem;
    box-shadow: 0 16px 32px rgba(0, 0, 0, 0.08);
  }

  .campus-map-frame iframe {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border: 0;
    border-radius: 1.3rem;
  }

  @media (max-width: 576px) {
    .campus-card {
      padding: 1.6rem 1.4rem;
    }

    .campus-map-frame {
      padding-bottom: 65%;
    }
  }

  .discussions-placeholder,
  .discussions-error {
    text-align: center;
    color: var(--text-muted);
    padding: 2rem 1rem;
  }

  .people-section {
    background-color: rgba(0, 0, 0, 0.02);
    border-radius: 1.5rem;
    padding: 2.5rem 2rem;
  }

  .people-section .section-title {
    text-align: center;
    margin-bottom: 2.5rem;
  }

  .people-grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 2.5rem;
  }

  .people-card {
    width: 220px;
    text-align: center;
    position: relative;
  }

  .people-card a {
    color: inherit;
    text-decoration: none;
  }

  .people-avatar {
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

  .people-name {
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 0.35rem;
  }

  .people-role {
    color: var(--text-muted);
    margin-bottom: 0.35rem;
  }

  .people-contact {
    color: var(--text-muted);
    font-size: 0.9rem;
    word-break: break-word;
  }

  @media (max-width: 576px) {
    .news-card,
    .people-card {
      width: 220px;
      padding: 1.75rem 1.25rem;
    }

    .people-avatar {
      width: 130px;
      height: 130px;
    }

    .hero-logo-card {
      width: 130px;
      padding: 0.85rem;
    }
  }
</style>

<section class="home-section hero-section mb-5">
  <div class="container px-0">
    <div class="row align-items-center g-5">
      <div class="col-lg-7">
        <div class="hero-badge">
          NLP @ Data Science and AI Division
        </div>
        <h1 class="hero-title">Interpretable, Controllable, Trustworthy Language Models</h1>
        <p class="hero-text mb-4">
          We conduct academic research in natural language processing, where we investigate methods to understand language models and how to apply them to the political and social sciences. We are part of the Data Science and AI division at <a href="https://www.chalmers.se/en/">Chalmers University of Technology</a>.
        </p>
      </div>
      <div class="col-lg-5">
        <div class="hero-logos">
          <a class="hero-logo-card" href="https://www.chalmers.se/en/" target="_blank" rel="noopener">
            <img src="assets/logos/chalmers.png" alt="Chalmers University of Technology">
          </a>
          <a class="hero-logo-card" href="https://www.gu.se/en" target="_blank" rel="noopener">
            <img src="assets/logos/gu.png" alt="University of Gothenburg">
          </a>
        </div>
      </div>
    </div>
  </div>
</section>

{% assign recent_news = site.news | sort: "date" | reverse %}
{% if recent_news.size > 0 %}
<section class="home-section py-5 news-section mb-5">
  <div class="section-title">
    <h2 class="h3 mb-1">Latest News</h2>
    <p class="text-muted mb-0">Highlights and updates from the group</p>
  </div>
  <div class="news-grid">
    {% for item in recent_news limit: 3 %}
    <article class="news-card">
      <a href="{{ item.url | relative_url }}">
        <h3>{{ item.title }}</h3>
        {% if item.date %}
        <div class="news-date">{{ item.date | date: "%b %-d, %Y" }}</div>
        {% endif %}
        {% if item.excerpt %}
        <p class="news-excerpt">{{ item.excerpt | strip_html | truncate: 160 }}</p>
        {% endif %}
      </a>
    </article>
    {% endfor %}
  </div>
</section>
{% endif %}

<section class="home-section py-5 discussions-section mb-5">
  <div class="section-title">
    <h2 class="h3 mb-1">Meeting Announcements</h2>
    <p class="text-muted mb-0">Live updates from our GitHub Discussions board</p>
  </div>
  <div class="discussions-grid" id="discussions-feed">
    <div class="discussions-placeholder">Loading latest discussions…</div>
  </div>
</section>

{% assign active_members = site.members | where: "state", "current" | sort: "name" %}
{% assign staff_members = "" | split: "" %}
{% assign student_members = "" | split: "" %}
{% assign other_members = "" | split: "" %}
{% for member in active_members %}
  {% assign position = member.position | downcase %}
  {% if position contains "professor" or position contains "lecturer" or position contains "director" %}
    {% assign staff_members = staff_members | push: member %}
  {% elsif position contains "phd" or position contains "student" or position contains "intern" %}
    {% assign student_members = student_members | push: member %}
  {% else %}
    {% assign other_members = other_members | push: member %}
  {% endif %}
{% endfor %}

{% if staff_members.size > 0 or student_members.size > 0 or other_members.size > 0 %}
<section class="home-section py-5 people-section mb-5">
  <div class="section-title">
    <h2 class="h3 mb-1">People</h2>
    <p class="text-muted mb-0">Active members of the NLP@DSAI group</p>
  </div>

  {% if staff_members.size > 0 %}
  <div class="mb-5">
    <h3 class="h4 text-center mb-4">Staff</h3>
    <div class="people-grid">
      {% for member in staff_members %}
      <div class="people-card">
        <a href="{{ member.url | relative_url }}">
          <img class="people-avatar" src="{{ member.image | default: 'assets/members/placeholder.png' | relative_url }}" alt="{{ member.name }}">
          <div class="people-name">{{ member.name }}</div>
          {% if member.position %}
          <div class="people-role">{{ member.position }}</div>
          {% endif %}
          {% if member.email %}
          <div class="people-contact">{{ member.email }}</div>
          {% endif %}
        </a>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  {% if student_members.size > 0 %}
  <div class="mb-5">
    <h3 class="h4 text-center mb-4">Students</h3>
    <div class="people-grid">
      {% for member in student_members %}
      <div class="people-card">
        <a href="{{ member.url | relative_url }}">
          <img class="people-avatar" src="{{ member.image | default: 'assets/members/placeholder.png' | relative_url }}" alt="{{ member.name }}">
          <div class="people-name">{{ member.name }}</div>
          {% if member.position %}
          <div class="people-role">{{ member.position }}</div>
          {% endif %}
          {% if member.email %}
          <div class="people-contact">{{ member.email }}</div>
          {% endif %}
        </a>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}

  {% if other_members.size > 0 %}
  <div>
    <h3 class="h4 text-center mb-4">Team</h3>
    <div class="people-grid">
      {% for member in other_members %}
      <div class="people-card">
        <a href="{{ member.url | relative_url }}">
          <img class="people-avatar" src="{{ member.image | default: 'assets/members/placeholder.png' | relative_url }}" alt="{{ member.name }}">
          <div class="people-name">{{ member.name }}</div>
          {% if member.position %}
          <div class="people-role">{{ member.position }}</div>
          {% endif %}
          {% if member.email %}
          <div class="people-contact">{{ member.email }}</div>
          {% endif %}
        </a>
      </div>
      {% endfor %}
    </div>
  </div>
  {% endif %}
</section>
{% endif %}

<section class="home-section py-5 funding-section mb-5">
  <div class="section-title">
    <h2 class="h3 mb-1">We Are Gratefully Funded By</h2>
    <p class="text-muted mb-0">The organisations that make our research possible</p>
  </div>
  <div class="funding-logos">
    <div class="funding-logo-card">
      <img src="assets/logos/wasp.png" alt="Wallenberg AI, Autonomous Systems and Software Program">
      <div class="funding-caption">Wallenberg AI, Autonomous Systems and Software Program (WASP)</div>
    </div>
    <div class="funding-logo-card">
      <img src="assets/logos/wasp-hs.png" alt="WASP-HS">
      <div class="funding-caption">Wallenberg AI and Autonomous Systems Program – Humanities and Society (WASP-HS)</div>
    </div>
    <div class="funding-logo-card">
      <img src="assets/logos/chalmers.png" alt="Chalmers University of Technology">
      <div class="funding-caption">Chalmers University of Technology</div>
    </div>
    <div class="funding-logo-card">
      <img src="assets/logos/gu.png" alt="University of Gothenburg">
      <div class="funding-caption">University of Gothenburg</div>
    </div>
  </div>
</section>

<section class="home-section py-5">
  <div class="campus-section">
    <div class="section-title">
      <h2 class="h3 mb-3">Find Us on Campus</h2>
      <p class="text-muted mb-0">Johanneberg campus · Department of Computer Science and Engineering</p>
    </div>
    <div class="campus-grid">
      <div class="campus-card">
        <h3>Visit NLP@DSAI</h3>
        <ul>
          <li><strong>Address</strong><span>Go to building EDIT trappa D, E och F. Entrance from Rännvägen 6. Go to Floor 5. Room 5478</span></li>
          <li><strong>Questions</strong><span>Email the organisers via <a href="mailto:richard.johansson@gu.se">richard.johansson@gu.se</a></span></li>
        </ul>
      </div>
      <div class="campus-map-frame">
        <iframe
          src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d485.54515034376004!2d11.977766830051518!3d57.687687740709144!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x464ff30b01337d59%3A0x1180b1d57c06ca10!2sEDIT%20Building!5e0!3m2!1sen!2sse!4v1761439811754!5m2!1sen!2sse"
          allowfullscreen=""
          loading="lazy"
          referrerpolicy="no-referrer-when-downgrade">
        </iframe>
      </div>
    </div>
  </div>
</section>

<script>
  document.addEventListener('DOMContentLoaded', function () {
    const feedContainer = document.getElementById('discussions-feed');
    if (!feedContainer) {
      return;
    }

    const DISCUSSIONS_FEED_URL = 'https://github.com/orgs/dsai-nlp/discussions/categories/meetings-announcements.atom';
    const proxyFetch = (url) => {
      const proxied = `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`;
      return fetch(proxied);
    };

    const parseHtmlSnippet = (htmlString) => {
      const temp = document.createElement('div');
      temp.innerHTML = htmlString;
      return temp.textContent || temp.innerText || '';
    };

    const renderEntries = (entries) => {
      if (!entries.length) {
        feedContainer.innerHTML = '<div class="discussions-placeholder">No recent discussions yet. Check back soon!</div>';
        return;
      }

      feedContainer.innerHTML = '';
      entries.forEach((entry) => {
        const titleEl = entry.querySelector('title');
        const title = titleEl ? titleEl.textContent.trim() : 'Untitled discussion';

        const linkEl = Array.from(entry.getElementsByTagName('link')).find((link) => {
          const rel = link.getAttribute('rel');
          return !rel || rel === 'alternate';
        });
        const href = linkEl ? linkEl.getAttribute('href') : '#';

        const updatedEl = entry.querySelector('updated');
        const updatedDate = updatedEl ? new Date(updatedEl.textContent) : null;
        const formattedDate = updatedDate && !isNaN(updatedDate) ? updatedDate.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' }) : 'Recently';

        const summaryEl = entry.querySelector('summary');
        let preview = summaryEl ? parseHtmlSnippet(summaryEl.textContent).trim() : '';
        if (preview.length > 200) {
          preview = preview.slice(0, 197).trimEnd() + '…';
        }

        const details = extractContentDetails(entry);
        const detailsMarkup = details.length
          ? `<ul class="discussion-details">${details
              .map(
                (detail) => `
                  <li class="discussion-detail">
                    <strong>${detail.label}:</strong> ${detail.text}
                  </li>
                `
              )
              .join('')}</ul>`
          : '';

        const card = document.createElement('a');
        card.className = 'discussion-card';
        card.href = href;
        card.target = '_blank';
        card.rel = 'noopener noreferrer';
        card.innerHTML = `
          <div class="discussion-title">${title}</div>
          <div class="discussion-meta">Updated ${formattedDate}</div>
          ${detailsMarkup || (preview ? `<div class="discussion-preview">${preview}</div>` : '')}
        `;
        feedContainer.appendChild(card);
      });
    };

    const extractFallbackDate = (entry) => {
      const publishedEl = entry.querySelector('published');
      const updatedEl = entry.querySelector('updated');
      const raw = publishedEl ? publishedEl.textContent : updatedEl ? updatedEl.textContent : null;
      if (!raw) {
        return null;
      }
      const parsed = Date.parse(raw);
      return Number.isNaN(parsed) ? null : new Date(parsed);
    };

    const extractContentDetails = (entry) => {
      const contentEl = entry.querySelector('content');
      if (!contentEl) {
        return [];
      }

      const temp = document.createElement('div');
      temp.innerHTML = contentEl.textContent;

      const sections = [];
      const headings = Array.from(temp.querySelectorAll('h3, h2'));

      headings.forEach((heading) => {
        const labelRaw = heading.textContent.replace(/[:：]/g, '').trim();
        const labelClean = labelRaw.replace(/^[^\p{L}\p{N}]+/u, '').trim();
        const labelLower = labelClean.toLowerCase();
        let sibling = heading.nextElementSibling;
        while (sibling && sibling.nodeType === Node.TEXT_NODE) {
          sibling = sibling.nextSibling;
        }

        if (!sibling || sibling.tagName.toLowerCase() !== 'p') {
          return;
        }

        const text = parseHtmlSnippet(sibling.innerHTML).trim();
        if (!text) {
          return;
        }

        sections.push({
          label: labelClean || labelRaw,
          text
        });
      });

      return sections;
    };

    const extractEventDate = (entry, fallbackDate) => {
      const contentEl = entry.querySelector('content');
      if (!contentEl) {
        return null;
      }

      const temp = document.createElement('div');
      temp.innerHTML = contentEl.textContent;

      const headings = Array.from(temp.querySelectorAll('h3, h2'));
      let dateParagraph = null;
      for (const heading of headings) {
        if (/date/i.test(heading.textContent)) {
          let sibling = heading.nextElementSibling;
          while (sibling && sibling.nodeType === Node.TEXT_NODE) {
            sibling = sibling.nextSibling;
          }
          if (sibling && sibling.tagName && sibling.tagName.toLowerCase() === 'p') {
            dateParagraph = sibling;
            break;
          }
        }
      }

      if (!dateParagraph) {
        return null;
      }

      let dateText = dateParagraph.textContent
        .replace(/[📆🗓️]/g, '')
        .replace(/\bon\b/gi, '')
        .replace(/\bof\b/gi, '')
        .replace(/\sat\b/gi, '')
        .replace(/\s+/g, ' ')
        .trim();

      if (!dateText) {
        return null;
      }

      const fallbackYear = fallbackDate ? fallbackDate.getFullYear() : new Date().getFullYear();
      if (!/\d{4}/.test(dateText)) {
        dateText += ` ${fallbackYear}`;
      }

      dateText = dateText
        .replace(/(\d+)(st|nd|rd|th)/gi, '$1')
        .replace(/(\d{1,2}:\d{2})\s*[-–]\s*\d{1,2}:\d{2}/, '$1');

      let parsedTime = Date.parse(dateText);

      if (Number.isNaN(parsedTime) && fallbackDate) {
        const monthDayMatch = dateText.match(/([A-Za-z]+)\s+(\d{1,2})/);
        if (monthDayMatch) {
          const month = monthDayMatch[1];
          const day = monthDayMatch[2];
          const timeMatch = dateText.match(/(\d{1,2}):(\d{2})/);
          const hour = timeMatch ? parseInt(timeMatch[1], 10) : 9;
          const minute = timeMatch ? parseInt(timeMatch[2], 10) : 0;
          parsedTime = Date.parse(`${month} ${day}, ${fallbackYear} ${hour}:${minute}`);
        }
      }

      if (Number.isNaN(parsedTime)) {
        return fallbackDate || null;
      }

      return new Date(parsedTime);
    };

    const fetchFeed = (url, useProxy = false) => {
      const request = useProxy ? proxyFetch(url) : fetch(url);
      return request.then((response) => {
        if (!response.ok) {
          if (!useProxy) {
            throw new Error('retry-with-proxy');
          }
          throw new Error('Failed to fetch discussions feed');
        }
        return response.text();
      });
    };

    fetchFeed(DISCUSSIONS_FEED_URL)
      .catch((error) => {
        if (error.message === 'retry-with-proxy' || error.name === 'TypeError') {
          return fetchFeed(DISCUSSIONS_FEED_URL, true);
        }
        throw error;
      })
      .then((xmlText) => {
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(xmlText, 'application/xml');

        if (xmlDoc.querySelector('parsererror')) {
          throw new Error('Unable to parse discussions feed');
        }

        const rawEntries = Array.from(xmlDoc.getElementsByTagName('entry'))
          .filter((entry) => {
            const categories = Array.from(entry.getElementsByTagName('category'));
            return !categories.some((cat) => {
              const term = (cat.getAttribute('term') || '').toLowerCase();
              return term.includes('state:closed') || term === 'closed';
            });
          });

        const now = new Date();
        const openEntries = [];

        for (const entry of rawEntries) {
          if (openEntries.length >= 3) {
            break;
          }

          const fallbackDate = extractFallbackDate(entry);
          const eventDate = extractEventDate(entry, fallbackDate);
          if (eventDate && eventDate < now) {
            continue;
          }
          openEntries.push(entry);
        }

        renderEntries(openEntries);
      })
      .catch((_error) => {
        feedContainer.innerHTML = `
          <div class="discussions-error">
            We couldn&#39;t load the discussions right now.
            <br>
            <a href="https://github.com/orgs/dsai-nlp/discussions/categories/meetings-announcements" target="_blank" rel="noopener noreferrer">
              View announcements on GitHub instead.
            </a>
          </div>
        `;
      });
  });
</script>
