import os
import requests
import time
import pandas as pd
import json
from datetime import datetime

GITHUB_USERNAME = "henryzhan013"
BASE_URL = "https://api.github.com"
OUTPUT_DIR = "data"

def get_headers(token):
    return{
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

def make_request(url, headers, params=None):
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 403:
        print("  Rate limited! Waiting 60 seconds...")
        time.sleep(60)
        return make_request(url, headers, params)
    else:
        print(f" Error{response.status_code}: {response.text[:200]}")
        return None

def get_profile(headers):
    print("\n👤 Fetching profile...")
    data = make_request(f"{BASE_URL}/users/{GITHUB_USERNAME}",headers)
    if data:
        return {
            "username": data.get("login",""),
            "name": data.get("name",""),
            "bio": data.get("bio",""),
            "company": data.get("company",""),
            "location": data.get("location",""),
            "public_repos": data.get("public_repos",0),
            "followers": data.get("followers", 0),
            "following": data.get("following", 0),
            "created_at": data.get("created_at",""),
        }
    return {}

def get_repos(headers):
    print("\n📦 Fetching repositories...")
    repos = []
    page = 1
    while True:
        data = make_request(
            f"{BASE_URL}/users/{GITHUB_USERNAME}/repos",
            headers,
            params={"per_page": 100, "page":page, "type": "all"}
        )
        if not data:
            break
        repos.extend(data)
        if len(data) < 100:
            break
        page += 1
    
    print(f"  Found {len(repos)} repositories")
    repos_clean = []
    for r in repos:
        repos_clean.append({
            "repo_name": r["name"],
            "full_name": r["full_name"],
            "description": r.get("description", ""),
            "language": r.get("language", ""),
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "pushed_at": r.get("pushed_at", ""),
            "size_kb": r["size"],
            "stargazers_count": r["stargazers_count"],
            "watchers_count": r["watchers_count"],
            "forks_count": r["forks_count"],
            "open_issues_count": r["open_issues_count"],
            "is_fork": r["fork"],
            "is_private": r["private"],
            "default_branch": r.get("default_branch", "main"),
            "topics": ", ".join(r.get("topics", [])),
            "html_url": r["html_url"],
        })
    
    return repos_clean, repos

def get_commits_for_repo(repo_name, headers):
    """Get all commits for a specific repo."""
    commits = []
    page = 1
    while True:
        data = make_request(
            f"{BASE_URL}/repos/{GITHUB_USERNAME}/{repo_name}/commits",
            headers,
            params={"per_page": 100, "page": page, "author": GITHUB_USERNAME}
        )
        if not data:
            break
        commits.extend(data)
        if len(data) < 100:
            break
        page += 1
    return commits

def get_all_commits(repos_raw, headers):
    print("\n💬 Fetching commits for each repo...")
    all_commits = []
    for repo in repos_raw:
        repo_name = repo["name"]
        print(f"  Fetching commits for {repo_name}...")
        commits = get_commits_for_repo(repo_name, headers)

        for c in commits:
            commit_data = c.get("commit", {})
            author = commit_data.get("author", {})
            all_commits.append({
                "repo_name": repo_name,
                "sha": c["sha"],
                "message": commit_data.get("message", ""),
                "author_name": author.get("name", ""),
                "author_email": author.get("email", ""),
                "date": author.get("date", ""),
                "html_url": c.get("html_url", ""),
            })
        time.sleep(0.5)  # be nice to the API
    
    print(f"  Found {len(all_commits)} total commits")
    return all_commits

def get_languages_for_repo(repo_name, headers):
    """Get language breakdown for a repo."""
    data = make_request(
        f"{BASE_URL}/repos/{GITHUB_USERNAME}/{repo_name}/languages",
        headers
    )
    return data if data else {}

def get_all_languages(repos_raw, headers):
    """Get language data across all repos."""
    print("\n🔤 Fetching language breakdowns...")
    all_languages = []
    
    for repo in repos_raw:
        repo_name = repo["name"]
        langs = get_languages_for_repo(repo_name, headers)
        
        for lang, bytes_count in langs.items():
            all_languages.append({
                "repo_name": repo_name,
                "language": lang,
                "bytes": bytes_count,
            })
        
        time.sleep(0.3)
    
    print(f"  Found language data for {len(set(r['repo_name'] for r in all_languages))} repos")
    return all_languages

def get_readme_for_repo(repo_name, default_branch, headers):
    """Get README content for a repo."""
    data = make_request(
        f"{BASE_URL}/repos/{GITHUB_USERNAME}/{repo_name}/readme",
        headers
    )
    if data and "content" in data:
        import base64
        try:
            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            return content
        except Exception:
            return ""
    return ""

def get_all_readmes(repos_clean, headers):
    """Get README files for all repos."""
    print("\n📄 Fetching README files...")
    readmes = []
    
    for repo in repos_clean:
        repo_name = repo["repo_name"]
        content = get_readme_for_repo(repo_name, repo["default_branch"], headers)
        if content:
            readmes.append({
                "repo_name": repo_name,
                "readme_content": content,
                "readme_length": len(content),
            })
            print(f"  ✓ {repo_name} ({len(content)} chars)")
        else:
            print(f"  ✗ {repo_name} (no README)")
        
        time.sleep(0.3)
    
    print(f"  Found {len(readmes)} READMEs")
    return readmes

def get_events(headers):
    """Get recent public events (pushes, PRs, issues, etc.)."""
    print("\n📡 Fetching recent events...")
    events = []
    page = 1
    while page <= 3:  # events API only goes back ~90 days anyway
        data = make_request(
            f"{BASE_URL}/users/{GITHUB_USERNAME}/events",
            headers,
            params={"per_page": 100, "page": page}
        )
        if not data:
            break
        events.extend(data)
        if len(data) < 100:
            break
        page += 1
    
    events_clean = []
    for e in events:
        events_clean.append({
            "type": e.get("type", ""),
            "repo_name": e.get("repo", {}).get("name", ""),
            "created_at": e.get("created_at", ""),
            "payload_action": e.get("payload", {}).get("action", ""),
        })
    
    print(f"  Found {len(events_clean)} recent events")
    return events_clean


def main():
    print("=" * 60)
    print("  GitHub Data Collector for Self-Analysis Project")
    print("=" * 60)

    token = input("\nPaste your GitHub token here: ").strip()
    if not token:
        print("No token provided. Exiting.")
        return
    
    headers = get_headers(token)

    os.makedirs(OUTPUT_DIR, exist_ok = True)

    profile = get_profile(headers)
    repos_clean, repos_raw = get_repos(headers)
    commits = get_all_commits(repos_raw, headers)
    languages = get_all_languages(repos_raw, headers)
    readmes = get_all_readmes(repos_clean, headers)
    events = get_events(headers)

    print("\n💾 Saving data to CSV files...")
    
    pd.DataFrame([profile]).to_csv(f"{OUTPUT_DIR}/profile.csv", index=False)
    print(f"  ✓ {OUTPUT_DIR}/profile.csv")
    
    pd.DataFrame(repos_clean).to_csv(f"{OUTPUT_DIR}/repos.csv", index=False)
    print(f"  ✓ {OUTPUT_DIR}/repos.csv")
    
    pd.DataFrame(commits).to_csv(f"{OUTPUT_DIR}/commits.csv", index=False)
    print(f"  ✓ {OUTPUT_DIR}/commits.csv")
    
    pd.DataFrame(languages).to_csv(f"{OUTPUT_DIR}/languages.csv", index=False)
    print(f"  ✓ {OUTPUT_DIR}/languages.csv")
    
    pd.DataFrame(readmes).to_csv(f"{OUTPUT_DIR}/readmes.csv", index=False)
    print(f"  ✓ {OUTPUT_DIR}/readmes.csv")
    
    pd.DataFrame(events).to_csv(f"{OUTPUT_DIR}/events.csv", index=False)
    print(f"  ✓ {OUTPUT_DIR}/events.csv")

        # Also save raw JSON for reference
    with open(f"{OUTPUT_DIR}/raw_repos.json", "w") as f:
        json.dump(repos_raw, f, indent=2)
    print(f"  ✓ {OUTPUT_DIR}/raw_repos.json")
    
    # Summary
    print("\n" + "=" * 60)
    print("  ✅ Data collection complete!")
    print("=" * 60)
    print(f"  Repos:    {len(repos_clean)}")
    print(f"  Commits:  {len(commits)}")
    print(f"  READMEs:  {len(readmes)}")
    print(f"  Events:   {len(events)}")
    print(f"\n  All files saved to ./{OUTPUT_DIR}/")
    print("  You're ready for analysis! 🚀")




if __name__ == "__main__":
    main()
