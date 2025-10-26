  # DSAI-NLP GitHub Page Contribution Guide

Welcome to the DSAI-NLP GitHub repository! This guide will walk you through the steps to contribute to our GitHub Page. Whether you're fixing a bug, adding content, or updating existing information, your contributions are invaluable to the success of our project.

## Important Note for Contributors

When creating pull requests, please ensure you are submitting them to the correct base repository, which is `dsai-nlp/dsai-nlp.github.io`, **not** the original `alshedivat/al-folio` repo from which this project was forked. You can select the base repository when you open a pull request on GitHub. For more details, please refer to our contribution guidelines.

## Local Development Using Docker

For ease of setup and to avoid compatibility issues with Ruby and Jekyll versions, we recommend using Docker for local development. This method isolates the environment and works consistently across different operating systems.

### Prerequisites

Before you begin, ensure you have Docker installed on your system. If you haven't already, you can download and install Docker from the following links:

- [Get Docker](https://docs.docker.com/get-docker/)
- [Install Docker Compose](https://docs.docker.com/compose/install/)

### Running the Website Locally

After installing Docker and Docker Compose, follow these steps to run the website on your local machine:

1. Open a terminal and navigate to the root directory of your cloned version of the `dsai-nlp/dsai-nlp.github.io` repository.

2. Run the following command to pull the latest pre-built Jekyll image from DockerHub:

   ```bash
   docker compose up --build
   ```

This command will start a local web server running the website.

4. Open your web browser and go to `http://localhost:8080`. You should see the local version of the website running.

Now that you have the site running locally, you can begin customizing the theme to suit your needs.

## Development Workflow

We follow [**GitHub Flow**](https://docs.github.com/en/get-started/using-github/github-flow) for continuous integration and deployment.

### 1. `master` is always stable

Never commit directly to `master`. All updates go through **Pull Requests (PRs)**.

```bash
git checkout master
git pull origin master
```

### 2. Create a feature branch

Each task or feature gets its own branch.

```bash
git checkout -b feature/add-content
# Make and commit changes
git add .
git commit -m "Add new content"
git push origin feature/add-content
```

### 3. Open a Pull Request

Use GitHub's web interface:

1. Go to the repository page.
2. Click **"Compare & pull request"**.
3. Add a title and short description.
4. Assign reviewers or labels if needed.
5. Click **"Create pull request."**

---

### 4. Review, test, and merge

Wait for automated checks (if configured) to pass, then request reviews.

- Reviewers may comment or approve.
- Once approved, click **"Merge pull request."**
- Use **Squash and merge** to keep a clean commit history.

### 5. Clean up merged branches

After merging, delete the feature branch:

```bash
# In GitHub UI
Click "Delete branch"

# Locally
git branch -d feature/add-content
git fetch -p
```

### 6. Deploy from `master`

The `master` branch always represents what is live in production.


## Quick Start

1. **Clone or fork the repo**
   ```bash
   git clone git@github.com:dsai-nlp/dsai-nlp.github.io.git
   cd dsai-nlp.github.io
   ```
   Fork first if you do not have push access.

2. **Create a feature branch**
   ```bash
   git checkout master
   git pull origin master
   git checkout -b feature/add-content
   ```

3. **Run locally (optional but recommended)**
   ```bash
   docker compose pull
   docker compose up
   ```
   Visit `http://localhost:8080`.

4. **Make changes, commit, and push**
   ```bash
   git add .
   git commit -m "Describe your change"
   git push origin feature/add-content
   ```

5. **Open a pull request** targeting `master`, add a short summary, and request a review.

## Pull Request Review

Once your pull request is submitted, it will be reviewed by the maintainers. If everything looks good, your changes will be merged into the `master` branch. If there are additional changes needed, feedback will be provided directly on the PR.

## Additional Guidelines

- Keep changes focused and follow the repository coding/content conventions.
- Test locally before opening a PR.
- Never push directly to `master`; always work on a branch.

## Extended Contribution Guides

The following sections provide more detailed instructions for specific content types. Refer to these when you need to add or update structured data on the site.

### Adding Your Profile

To add your profile to the NLP@DSAI team page:

1. Navigate to the `_members` directory.
2. Create a new Markdown file (`.md`) with a suitable name, e.g., `john_doe.md`.
3. Use the following template as a guide for your content. Fill in your personal details where applicable:

```yaml
---
name: Your Name
image: path_to_your_image
position: Your Position
state: current # Options: current, master, alumni
start-date: YYYY-MM-DD
end-date: YYYY-MM-DD # If applicable
pronouns: your pronouns
email: your_email@domain.com
# Add any of the following that apply to you:
scholar_userid:
publons_id:
research_gate_profile:
github_username:
linkedin_username:
twitter_username:
medium_username:
quora_username:
blogger_url:
work_url:
wikidata_id:
strava_userid:
keybase_username:
gitlab_username:
dblp_url:
description: "A short personal bio or description."
---
```

### Adding an Event

The events will be fetched from our public Github Discussion, so there's no need to set each event anymore.


### Adding an Announcement

To add an announcement:

1. Navigate to the `_news` directory.
2. Create a new Markdown file (`.md`) with a suitable name, e.g., `conference_announcement.md`.
3. Follow the predefined template in the `_news` folder for the announcement, filling in the relevant details.

### Adding or Updating Course Information

To add or update information about a course:

1. Navigate to the `_pages` directory.
2. Locate the `courses.md` file.
3. Update the file with the new course information or make changes to the existing content. You can find those files inside the directories in `_pages`.

### Updating Publications

To add or update publications:

1. Navigate to the `_bibliography` directory.
2. Locate the `papers.bib` file.
3. Add your new publication details in BibTeX format or update existing entries as necessary.

## Final Steps

After making any of the above changes, follow the standard contribution workflow:

1. Commit your changes with a clear message.
2. Push the changes to your feature branch.
3. Create a pull request to the main repository: `dsai-nlp/dsai-nlp.github.io` targeting `master`.

The repository maintainers will review your pull request and merge it upon approval. Make sure to provide detailed information in your pull request to expedite the review process.
