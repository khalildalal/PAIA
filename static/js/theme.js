(function () {
  const root = document.documentElement;
  const initial = window.APP_THEME || localStorage.getItem("theme_preference") || "system";

  function setTheme(theme, shouldReload) {
    const nextTheme = ["light", "dark", "system"].includes(theme) ? theme : "system";
    root.setAttribute("data-theme", nextTheme);
    localStorage.setItem("theme_preference", nextTheme);
    document.cookie = "theme_preference=" + nextTheme + "; path=/; max-age=31536000";

    document.querySelectorAll(".theme-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.themeChoice === nextTheme);
    });

    if (shouldReload) {
      window.location.reload();
    }
  }

  document.addEventListener("click", function (event) {
    const btn = event.target.closest(".theme-btn");
    if (!btn) return;

    const chosenTheme = btn.dataset.themeChoice;
    const currentTheme = root.getAttribute("data-theme") || "system";

    if (chosenTheme === currentTheme) return;

    setTheme(chosenTheme, true);
  });

  setTheme(initial, false);
})();