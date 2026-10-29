# Chakaso website

A lightweight, responsive static website for the Chakaso language-model research project. It uses plain HTML, CSS and JavaScript: no build step, framework, API key or backend required.

## Preview locally

Open `index.html` in a browser, or run a local static server from the repository root:

```bash
python -m http.server 8000
```

Then visit `http://localhost:8000/website/`.

## Deploy with Cloudflare Pages

1. Sign in to Cloudflare and open **Workers & Pages**.
2. Create an application → **Pages** → connect the GitHub repository `bhogesararam23/chakaso`.
3. Configure the production branch as `main`.
4. Set the root directory to `website`.
5. Leave the build command empty; use `/` as the build output directory (the root directory is already `website`).
6. Deploy and test the generated `*.pages.dev` URL.
7. In the Pages project, open **Custom domains** → **Set up a custom domain** and add `chakaso.tech`. Add `www.chakaso.tech` too if you want it.
8. Follow the exact DNS instructions Cloudflare shows. DNS records differ by registrar and current DNS provider; do not delete existing mail or verification records. If Cloudflare asks you to move authoritative nameservers, do that at the registrar only after reviewing all existing DNS records.

Cloudflare's dashboard and domain registrar require the account owner's sign-in and approval. Do not share passwords or authentication codes.

## Keep claims accurate

The site intentionally states that current checkpoints do not generate useful, coherent assistant responses. Update this wording only when new evaluation evidence supports a change. Keep benchmark claims linked to experiment records and the model card.
