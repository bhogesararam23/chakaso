const menuButton = document.querySelector(".menu-toggle");
const navigation = document.querySelector("nav");

menuButton?.addEventListener("click", () => {
  const isOpen = navigation?.classList.toggle("open") ?? false;
  menuButton.setAttribute("aria-expanded", String(isOpen));
});

navigation?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    navigation.classList.remove("open");
    menuButton?.setAttribute("aria-expanded", "false");
  });
});

const copyButton = document.querySelector("#copy-command");
copyButton?.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText("git clone https://github.com/bhogesararam23/chakaso.git");
    copyButton.textContent = "COPIED";
    window.setTimeout(() => { copyButton.textContent = "COPY"; }, 1600);
  } catch {
    copyButton.textContent = "COPY MANUALLY";
  }
});

const year = document.querySelector("#year");
if (year) year.textContent = String(new Date().getFullYear());