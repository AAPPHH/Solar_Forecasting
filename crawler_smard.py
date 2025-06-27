from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError

cwd = Path.cwd()

plant_label   = "Kraftwerk Neurath"
resolution_key = "quarterhour"
filetype_key   = "text/csv"

start_date = "01.06.2024"
end_date   = "02.06.2024"

download_dir = cwd / "downloads"

RESOLUTION_LABEL = {
    "quarterhour": "Auflösung: Viertelstunde",
    "hour":        "Auflösung: Stunde",
    "day":         "Auflösung: Tag",
    "week":        "Auflösung: Woche",
    "month":       "Auflösung: Monat",
    "year":        "Auflösung: Jahr",
}

FILETYPE_LABEL = {
    "text/csv": "CSV",
    "application/vnd.openxmlformats-officedocument.s": "XLSX",
    "text/xml": "XML",
}

resolution_label = RESOLUTION_LABEL[resolution_key]
filetype_label   = FILETYPE_LABEL[filetype_key]

download_dir.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, args=["--disable-gpu"])
    context = browser.new_context(accept_downloads=True,
                                  viewport={"width": 1920, "height": 1080})
    page = context.new_page()
    page.goto("https://www.smard.de/home/downloadcenter/download-kraftwerksdaten/", timeout=60_000)

    try:
        page.locator("button.js-cookie-accept").wait_for(state="visible", timeout=8000)
        page.locator("button.js-cookie-accept").click()
    except TimeoutError:
        pass

    page.locator("select[aria-label='Kraftwerk wählen']")\
        .select_option(label=plant_label)
    page.locator("select[aria-label='Auflösung wählen']")\
        .select_option(label=resolution_label)
    page.locator("select[aria-label='Dateiformat wählen']")\
        .select_option(label=filetype_label)

    frm = page.locator("input[name='daterange_from']")
    to  = page.locator("input[name='daterange_to']")

    frm.click()
    frm.fill(start_date)
    page.keyboard.press("Enter")

    to.click()
    to.fill(end_date)
    page.keyboard.press("Enter")

    with page.expect_download() as dl_info:
        page.locator("#help-powerplant-download").click()

    download   = dl_info.value
    final_path = download_dir / download.suggested_filename
    download.save_as(final_path)

    print(f"✅ Datei gespeichert unter: {final_path}")
    context.close()
    browser.close()