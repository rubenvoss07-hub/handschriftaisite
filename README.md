# Handschrift

Eine Webseite, die Text in Rubens Handschrift schreibt und als `.goodnotes`-Datei ausgibt. Alles
rechnet der Browser selbst (WebGPU, iPadOS 26 / Safari 26): die Seite lädt nur die Dateien hier.

- `model.enc`: die Modellgewichte, AES-256-GCM-verschlüsselt mit einem Schlüssel aus dem Passwort
  (PBKDF2-SHA256, 1.000.000 Runden). Ohne Passwort ist die Datei wertlos; das Passwort steht nirgends.
- `js/`: Generator, Lesemodelle, Auswahl, Layout und der GoodNotes-Schreiber in JavaScript (CPU-Referenz
  und WebGPU-Kernel).
- `test/`: Tests gegen PyTorch-Referenzwerte. Die Referenzdaten (`test/fixtures/`) entstehen im
  Forschungs-Repo mit `python web/test/dump.py`; dort steht auch, wie die Tests laufen.

Diese Dateien werden aus dem Forschungs-Repo (`handwriting-ai`, privat) mit `web/deploy.sh`
hierher veröffentlicht; Änderungen gehören dorthin, nicht hierher. Ein neues Modell: dort
`hw.py pack` ausführen und neu veröffentlichen.

Deploy: Vercel-Projekt mit diesem Repo (Framework: keins, Root: `/`), Vercel Authentication aus.
Oder GitHub Pages vom Branch `main`.
