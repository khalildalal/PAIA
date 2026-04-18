/*
Theme controller

This file controls the visual theme of the website.

What it does:
1. Reads the current theme preference
2. Saves the chosen theme in localStorage and cookies
3. Updates the page theme immediately
4. Reloads the page after a theme change so all pages stay visually consistent
*/

(function () {
  // Get the root <html> element so we can update its data-theme attribute.
  const root = document.documentElement;
  // Choose the initial theme using server state first, then saved browser state, then 'system'.
  const initial = window.APP_THEME || localStorage.getItem("theme_preference") || "system";

  // Apply the theme to the page and optionally refresh the page.
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

  // Listen for clicks on any theme button.
  document.addEventListener("click", function (event) {
    const btn = event.target.closest(".theme-btn");
    if (!btn) return;

    const chosenTheme = btn.dataset.themeChoice;
    const currentTheme = root.getAttribute("data-theme") || "system";

    if (chosenTheme === currentTheme) return;

    setTheme(chosenTheme, true);
  });

  // Apply the initial theme without reloading the page.
  setTheme(initial, false);
})();