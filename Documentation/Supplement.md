# Guidelines for Development Operations

**Contents:**

- [Guidelines for Development Operations](#guidelines-for-development-operations)
  - [Pre-requisites](#pre-requisites)
  - [Overview](#overview)
    - [Team Roles and Permissions](#team-roles-and-permissions)
    - [Trunk-based Development](#trunk-based-development)
  - [General Workflow](#general-workflow)
    - [Workflow Overview](#workflow-overview)
    - [Workflow in Detail](#workflow-in-detail)
  - [Creating a Release](#creating-a-release)

This guide describes general tools and workflows for using and maintaining a software project with versioning ([Development Operations](https://www.atlassian.com/devops/what-is-devops/how-to-start-devops)) for individual programmers and small teams alike.
The main goal of this guide is to promote clean repositories, maintainable code, and consistent project outputs while keeping repository maintenance as simple as possible.

Remember: **if in doubt, reach out!**

## Pre-requisites

- An integrated development environment (IDE) of your choice.
- A [Github](https://github.com/login) account
- [Git](https://git-scm.com/downloads) installed on your machine.
  - If you are new to Git, learn the basics [here](https://learn.microsoft.com/en-us/training/modules/intro-to-git/) (part of the [github foundations](https://learn.microsoft.com/en-us/training/paths/github-foundations/) course).
  - Refer to the supplementary [glossary](Supplement#glossary-of-terms) if you come across new Git-related jargon.
- Recommended: A Git GUI Client, such as [GitHub Desktop](https://desktop.github.com/) or [others](https://git-scm.com/downloads/guis).

## Overview

### Team Roles and Permissions

You will find yourself working on software projects either alone or in a small team. In the latter case, team roles may be fuzzy, but the team will generally contain **Juniors** and at least one **Senior**.

- **Juniors**: **commit** code, **submit** pull requests, **suggest** enhancements, **report** bugs, **create** issues (in the Github sense), and **consult** frequently with seniors.
- **Seniors** (Administrators): **review** code, **merge** pull requests, **report** to higher-ups, **supervise** project direction and coding standards, and **maintain** the repository. Crucially, they **consult** with the team and **provide guidance**.

For administrators: **Repository Management** can be enhanced by setting up a [Team](https://docs.github.com/en/organizations/organizing-members-into-teams) - this allows you to adjust access of teams members to your repository, protect your code, and use additional project management tools.

### Trunk-based Development

[Trunk-based](https://www.atlassian.com/continuous-delivery/continuous-integration/trunk-based-development) development is a version control practice designed around a main "trunk" branch with small, short-lived issue branches splitting from it and merging back into it.

This workflow is great for flexibility and collaboration. We will use a bare-bones version of this workflow to prioritize simplicity; additional practices typically associated with trunk-based workflows can be added in as needed.

Trunk-based development uses two branch types:

- **`main`**: stable state of the project, always assumed ready for deployment/release. The only persistent branch in the repository.
- **`issue`**: short-lived branches for work on new features, enhancements, bug fixes, etc. They branch off of `main`. They are merged back into `main` at the end of their lifespan. Delete them once they are successfully merged.
  - **NB**: Smaller `issue` branches (i.e., branches with few changes) are easier and faster to review, which promotes discussions and cleaner code.

## General Workflow

Remember: these are guidelines. The important part is that your software project has a **Github repository** with a **main branch** which contains **working code**. How you get there is up to you (and your team). The workflow suggested below was designed to keep things simple and clean while covering most common use-cases.

### Workflow Overview

1. Get **access** to an existing repo or create a new one
2. Create an **issue** (task description/bug report, etc.)
3. Create new **issue branch**
4. Make **local changes** to code
5. Push changes to the **remote**
6. Create a **pull request** once everything in the issue has been addressed
7. Review code, iterate, and discuss, and finally **merge** the pull request

The graphic below shows a generic trunk-based repository and the lifespan of one of its issue branches:

![Workflow illustrated](graphics/branch_lifespan.svg)

### Workflow in Detail

1. Create/Get Access to a Project Repository

   - You will typically get access from a senior.
   - *Seniors*:
     - Grant access to a repository in `Settings` → `Collaborators and teams` → `Add people`/[`Add teams`](https://docs.github.com/en/organizations/organizing-members-into-teams/about-teams)
     - Use the [template repository](https://github.com/CEITEC-CTLAB/template_repo.git) to quickly set up new repos
       - You can either go to the template and click `Use this template`
       - Or make a `New repository` and choose a template in the *Repository template* section
       - The linke template contains further instructions on how to set up your new repo
   - Once you have access **clone** the Repository to Your Local Machine
     - Use the [Command Line](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository), [GitHub Desktop](https://docs.github.com/en/desktop/adding-and-cloning-repositories/cloning-and-forking-repositories-from-github-desktop), or any other Git interface.

2. Create an Issue

   - Create a New Issue to Track Your Changes

     - `Issues` tab → `New issue`
     - Describe the issue. Start with an **issue template** if available, and add a descriptive title, a concise description, and appropriate labels.
     - Assign the issue to yourself or somebody else and `Submit`.

   - Setup Time/project Management for the Issue

     - Useful for more complex issues or larger teams. *Seniors*: use at your own discretion (see the [Supplement](Supplement#issue-time-and-project-management) for more info).

3. Create a New Issue Branch

   - This can be done on the issue page and [pulled](https://docs.github.com/en/desktop/working-with-your-remote-repository-on-github-or-github-enterprise/syncing-your-branch-in-github-desktop#pulling-to-your-local-branch-from-the-remote) or [created](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop#creating-a-branch) locally and then [pushed](https://docs.github.com/en/desktop/making-changes-in-a-branch/managing-branches-in-github-desktop#publishing-a-branch).
   - Choose a branch name, preferably one which follows the pattern: `<issue_ID>_short_description`
     - e.g.: `10-rng` - branch for issue #10 titled "adding a random number generator".

4. Make Changes to the Code Locally

   - Commit changes locally (using the GIT interface of your choice)
     - Commit frequently.
     - Change one thing per commit: Fix one bug or a group of related bugs, add one feature, etc.
     - Commit messages are [short and to the point](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) and written in the imperative:
       - Example: for issue with tag #10 in GitHub, commit a bugfix, where the function get_result() caused a DivisionByZero error for certain inputs:

         ```#10: Fix division by 0 in get_result()```

5. Push Changes to Remote GitHub Repository

   - Pushes can be *less frequent* than local commits, but important for *backup* and *sahring your code*.
   - Use [GitHub Desktop](https://docs.github.com/en/desktop/making-changes-in-a-branch/pushing-changes-to-github-from-github-desktop) or any other Git interface.

6. Create a Pull Request

   - After finishing your work and pushing all changes, *submit* a pull request (PR) from the `issue` branch to `main`.
     - `Pull requests` → `New pull request`
       - base branch: `base=main`, `compare=your_issue_branch`
     - Provide a title and concise description, which should explain the contents and purpose of the PR.
       - If the PR closes an issue (which may not always be the case), you can use [keywords](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue) to link the issue to the PR.
     - Assign responsible *seniors* who will `Review` and `Merge` your PR.
     - Click `Create pull request`.
     - **Notes**:
       - PRs are a place to discuss and improve code. Don't be afraid to open PRs and provide/receive constructive criticism.
       - Remember to **keep branches small!**
       - Before you open a PR, make sure to **rebase** your `issue` branch on the latest commit in `main`. This makes for much smoother merging.

7. Review and Merge

   - Wait for reviews of your pull request. Discuss your code, respond to feedback and requests for changes.
     - Do not hesitate to remind your reviewer if they're late with your review.
   - Once approved, your changes will be merged into the target branch by a *senior*.
     - There are several ways to merge an `issue` branch to `main`, with different use-cases. We recommend to:
       - **Rebase and merge** leads to a more granular and detailed commit history. Use if you want to preserve detail and/or if merging commits from multiple contributors.
       - **Squash and merge** leads to a leaner repository. Use if keeping every issue branch commit is not important.
     - **Delete** merged `issue` branch after merging

![Merging Options](graphics/merge_options.svg)

## Creating a Release

[Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) are packaged versions of software ready for distribution and use.
They are associated with a specific commit on the `main` branch using a [tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging).
The tag contains a unique version number, preferably using [*Calendar Versioning*](Supplement#calendar-versioning).
Releases can contain source code (included automatically), compiled binaries of the code, and any other relevant files (manuals, [documentation](templates/documentation/README.md), etc.).
To create a release:

1. In Github repository: `Releases` → `Draft a New Release`
2. `Choose a tag` → create a *CalVer* tag for the release (`vYY.MM.DD`) → `Create new tag...`
3. assign the `main` branch as the `Target`.
4. If applicable, choose a previous release tag to compare to the current release (auto is usually ok) and click `Generate release notes`
5. If needed, adjust the generated *release title* and *release notes*
6. Drop in any additional files you want to include in the release
7. At this point, you are ready to `Publish release`. Make sure to get the go-ahead from other team members beforehand.
