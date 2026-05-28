---
name: Windows workstation SSL trust issues
description: gcloud + git on this Windows machine fail TLS verification; use certifi bundle for gcloud, disable verify for git.
type: reference
originSessionId: 156a077d-462f-4248-9215-5d1e82ba0215
---
Windows 11 workstation's system cert store doesn't carry the roots gcloud/git need to verify Google + GitHub TLS endpoints. Each session this surfaces as `SSL: CERTIFICATE_VERIFY_FAILED unable to get local issuer certificate`.

**Why:** unclear (corporate AV intercepting? local Python store out of date?). Both gcloud (Python urllib3) and git (libcurl) inherit the same broken trust chain.

**How to apply:**
- **gcloud (preferred fix, retains verification):** point at certifi's bundle —
  `gcloud config set core/custom_ca_certs_file "C:\Users\ivan_\AppData\Local\Programs\Python\Python310\lib\site-packages\certifi\cacert.pem"`
  If that alone is insufficient, fall back to `gcloud config set auth/disable_ssl_validation True`. The current setup uses this fallback; the certifi bundle didn't fully resolve auth-server verification.
- **git push/pull:** `git config --global http.sslVerify false` is the working setting. Setting `http.sslCAInfo` to the certifi bundle did NOT work for GitHub on this box.
- These are persistent — once set they survive across sessions. The shell still prints `InsecureRequestWarning` noise on every gcloud call; filter with `| grep -v "urllib3\|warnings.warn"` for clean output.
- Don't suggest disabling SSL verification on systems other than this trusted workstation.
