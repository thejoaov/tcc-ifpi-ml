#!/usr/bin/env bash
set -euo pipefail

DECK="${1:-../apresentacao/index.html}"
DESTINO="${2:-./output/slides}"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LARGURA=960

[ -f "$DECK" ] || { echo "Deck não encontrado: $DECK" >&2; exit 1; }
[ -x "$CHROME" ] || { echo "Chrome não encontrado em $CHROME" >&2; exit 1; }
command -v magick >/dev/null || { echo "ImageMagick (magick) não encontrado" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$DESTINO"

# Cópia do deck que aceita ?slide=N e congela as animações, para o print sair no estado final.
python3 - "$DECK" "$TMP/deck.html" <<'PY'
import sys
html = open(sys.argv[1], encoding='utf-8').read()
injecao = """
<style>
  *, *::before, *::after { transition: none !important; animation: none !important; }
  .deck-counter, .edit-toggle, .edit-hotzone { display: none !important; }
</style>
<script>
  window.addEventListener('load', () => {
    const n = parseInt(new URLSearchParams(location.search).get('slide') || '1', 10);
    document.querySelectorAll('.slide').forEach((s, i) => {
      s.classList.toggle('active', i === n - 1);
      s.classList.toggle('visible', i === n - 1);
    });
  });
</script>
</body>"""
open(sys.argv[2], 'w', encoding='utf-8').write(html.replace('</body>', injecao, 1))
PY

cp -R "$(dirname "$DECK")/assets" "$TMP/" 2>/dev/null || true

TOTAL=$(grep -c '<section class="slide' "$DECK")
echo "Capturando $TOTAL slides de $DECK"

for n in $(seq 1 "$TOTAL"); do
	"$CHROME" --headless --disable-gpu --hide-scrollbars \
		--force-device-scale-factor=1 --window-size=1920,1080 \
		--virtual-time-budget=4000 \
		--screenshot="$TMP/cru.png" "file://$TMP/deck.html?slide=$n" >/dev/null 2>&1
	magick "$TMP/cru.png" -resize "${LARGURA}x" -strip \
		"$(printf '%s/slide-%02d.png' "$DESTINO" "$n")"
	printf '\r  %02d/%s' "$n" "$TOTAL"
done

echo ""
echo "Miniaturas em $DESTINO"
