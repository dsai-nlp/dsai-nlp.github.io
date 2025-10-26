---
layout: default
title: Events
permalink: /events
nav: true
nav_order: 2
---

<style>
  .events-hero {
    background: linear-gradient(135deg, rgba(10, 95, 166, 0.08), rgba(95, 28, 160, 0.08));
    border-radius: 2rem;
    padding: 3rem 2.5rem;
    margin-bottom: 3rem;
  }

  .events-hero h1 {
    font-size: clamp(2.2rem, 5vw, 3rem);
    font-weight: 700;
    margin-bottom: 0.75rem;
  }

  .events-hero p {
    font-size: 1.05rem;
    color: var(--text-muted);
    max-width: 720px;
    margin-bottom: 0;
  }

  .events-section {
    background-color: rgba(0, 0, 0, 0.02);
    border-radius: 1.5rem;
    padding: 2.5rem 2rem;
    margin-bottom: 3rem;
  }

  .events-section:last-of-type {
    margin-bottom: 0;
  }

  .events-section .section-title {
    text-align: center;
    margin-bottom: 2.5rem;
  }

  .events-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 2rem;
  }

  .event-card {
    display: block;
    padding: 2rem 1.6rem;
    border-radius: 1.4rem;
    background-color: var(--global-card-bg-color, #fff);
    box-shadow: 0 16px 32px rgba(0, 0, 0, 0.08);
    color: inherit;
    text-decoration: none;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }

  .event-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
  }

  .event-title {
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 0.45rem;
  }

  .event-meta {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-bottom: 0.9rem;
  }

  .event-details {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    font-size: 0.95rem;
    color: var(--global-text-color);
    word-break: break-word;
    overflow-wrap: anywhere;
  }

  .event-detail strong {
    font-weight: 600;
    margin-right: 0.35rem;
  }

  .events-placeholder,
  .events-error {
    text-align: center;
    color: var(--text-muted);
    padding: 2rem 1rem;
  }

  @media (max-width: 576px) {
    .events-hero {
      padding: 2.4rem 1.8rem;
    }

    .event-card {
      padding: 1.8rem 1.4rem;
    }
  }
</style>

<section class="events-hero">
  <h1>Stay in the Loop</h1>
  <p>We publish every meeting and social activity in the <a href="https://github.com/orgs/dsai-nlp/discussions/categories/meetings-announcements">Meetings & Events</a> board on GitHub Discussions, subscribe to the discussions board for real-time updates.</p>
</section>

<section class="events-section">
  <div class="section-title">
    <h2 class="h3 mb-1">Upcoming Events</h2>
  </div>
  <div class="events-grid" id="events-upcoming">
    <div class="events-placeholder">Loading upcoming events…</div>
  </div>
</section>

<section class="events-section">
  <div class="section-title">
    <h2 class="h3 mb-1">Archive</h2>
    <p class="text-muted mb-0">Recent highlights you might want to revisit</p>
  </div>
  <div class="events-grid" id="events-archive">
    <div class="events-placeholder">Loading archive…</div>
  </div>
</section>

<section class="events-section">
  <div class="section-title">
    <h2 class="h4 mb-2">Get Involved</h2>
    <p class="text-muted mb-0">Open a discussion thread to propose a topic, volunteer as a curator, or suggest a visiting speaker. We welcome demos, lightning talks, and interdisciplinary collabs.</p>
  </div>
</section>

<script>
  document.addEventListener('DOMContentLoaded', function () {
    const upcomingContainer = document.getElementById('events-upcoming');
    const archiveContainer = document.getElementById('events-archive');
    if (!upcomingContainer || !archiveContainer) {
      return;
    }

    const FEED_URL = 'https://github.com/orgs/dsai-nlp/discussions/categories/meetings-announcements.atom';
    const proxyFetch = (url) => {
      const proxied = `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`;
      return fetch(proxied);
    };

    const parseHtmlSnippet = (htmlString) => {
      const temp = document.createElement('div');
      temp.innerHTML = htmlString;
      return temp.textContent || temp.innerText || '';
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
        .replace(/\bat\b/gi, ' ')
        .replace(/\bof\b/gi, '')
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

    const extractDetails = (entry) => {
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
        const label = labelClean || labelRaw;

        if (!label) {
          return;
        }

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
          label,
          text
        });
      });

      return sections;
    };

    const buildCard = (entry, eventDate) => {
      const titleEl = entry.querySelector('title');
      const title = titleEl ? titleEl.textContent.trim() : 'Untitled event';

      const linkEl = Array.from(entry.getElementsByTagName('link')).find((link) => {
        const rel = link.getAttribute('rel');
        return !rel || rel === 'alternate';
      });
      const href = linkEl ? linkEl.getAttribute('href') : '#';

      const updatedEl = entry.querySelector('updated');
      const updatedDate = updatedEl ? new Date(updatedEl.textContent) : null;
      const formattedUpdated = updatedDate && !isNaN(updatedDate)
        ? updatedDate.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
        : 'Recently';

      const formattedEvent = eventDate
        ? eventDate.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
        : null;

      const details = extractDetails(entry);
      const detailsMarkup = details.length
        ? `<ul class="event-details">${details
            .map((detail) => `<li class="event-detail"><strong>${detail.label}:</strong> ${detail.text}</li>`)
            .join('')}</ul>`
        : '';

      const card = document.createElement('a');
      card.className = 'event-card';
      card.href = href;
      card.target = '_blank';
      card.rel = 'noopener noreferrer';
      card.innerHTML = `
        <div class="event-title">${title}</div>
        <div class="event-meta">
          ${formattedEvent ? `Happening on ${formattedEvent}` : 'Date TBA'} · Posted ${formattedUpdated}
        </div>
        ${detailsMarkup}
      `;
      return card;
    };

    const renderList = (container, entries, emptyMessage) => {
      if (!entries.length) {
        container.innerHTML = `<div class="events-placeholder">${emptyMessage}</div>`;
        return;
      }

      container.innerHTML = '';
      entries.forEach(({ entry, eventDate }) => {
        container.appendChild(buildCard(entry, eventDate));
      });
    };

    const fetchFeed = (url, useProxy = false) => {
      const request = useProxy ? proxyFetch(url) : fetch(url);
      return request.then((response) => {
        if (!response.ok) {
          if (!useProxy) {
            throw new Error('retry-with-proxy');
          }
          throw new Error('Failed to fetch events feed');
        }
        return response.text();
      });
    };

    fetchFeed(FEED_URL)
      .catch((error) => {
        if (error.message === 'retry-with-proxy' || error.name === 'TypeError') {
          return fetchFeed(FEED_URL, true);
        }
        throw error;
      })
      .then((xmlText) => {
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(xmlText, 'application/xml');

        if (xmlDoc.querySelector('parsererror')) {
          throw new Error('Unable to parse events feed');
        }

        const rawEntries = Array.from(xmlDoc.getElementsByTagName('entry'));
        const now = new Date();
        const upcoming = [];
        const past = [];

        rawEntries.forEach((entry) => {
          const fallbackDate = extractFallbackDate(entry);
          const eventDate = extractEventDate(entry, fallbackDate) || fallbackDate || new Date();
          const item = { entry, eventDate };

          if (eventDate >= now) {
            upcoming.push(item);
          } else {
            past.push(item);
          }
        });

        upcoming.sort((a, b) => a.eventDate - b.eventDate);
        past.sort((a, b) => b.eventDate - a.eventDate);

        renderList(upcomingContainer, upcoming.slice(0, 6), 'No upcoming events right now — check back soon!');
        renderList(archiveContainer, past.slice(0, 9), 'No archived events yet. New sessions will appear here.');
      })
      .catch((_error) => {
        const fallbackMarkup = `
          <div class="events-error">
            We couldn&#39;t load events right now.
            <br>
            <a href="https://github.com/orgs/dsai-nlp/discussions/categories/meetings-announcements" target="_blank" rel="noopener noreferrer">
              View announcements on GitHub instead.
            </a>
          </div>
        `;
        upcomingContainer.innerHTML = fallbackMarkup;
        archiveContainer.innerHTML = fallbackMarkup;
      });
  });
</script>
