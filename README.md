## Bybit Funding Rate Scanner Telegram Bot

Bybit linear perpetual kontratlarında **negatif funding** ve **son 5dk’da +%2+ yükseliş** kriterlerine uyan coinleri tarar; 09:00, 13:00, 17:00’de otomatik alarm + manuel `/scan` ile Telegram’dan bildirir. İsteğe bağlı ML ile **Long ihtimali %X** gösterilir.

### Gereksinimler

- Python 3.11+
- `.env` ile Telegram + Bybit ayarları

### Kurulum

1. Bağımlılıkları yükle:

```bash
pip install -r requirements.txt
```

2. `.env` oluştur (`.env.example` örneğine bak):

- `TELEGRAM_BOT_TOKEN` – BotFather token
- `TELEGRAM_CHAT_ID` – Bildirim gidecek chat ID
- `ADMIN_USER_ID` veya `ADMIN_USER_IDS` – Admin kullanıcı ID’leri (virgülle ayrılmış)
- `BYBIT_BASE_URL` – Mainnet: `https://api.bybit.com`
- İsteğe bağlı: `APP_TIMEZONE` (örn. `Asia/Chita`), `MIN_PRICE_CHANGE_PERCENT` (varsayılan 2.0)

3. Botu çalıştır:

```bash
python main.py
```

### ML Long/Short ihtimali (isteğe bağlı)

Bildirimde **Long ihtimali %X** satırı için Replit/ sunucuda bir kez:

```bash
python -m ml.dataset
python -m ml.dataset --train
```

Detay: `ml/README.md`

### Komutlar

- `/start` – Hoş geldin + komut listesi
- `/status` – Bot durumu, zaman dilimi, Bybit URL
- `/settings` – Bildirim ayarları
- `/scan` – Manuel tam tarama (admin)
- `/scan_rise` – Sadece yükseliş (admin)
- `/scan_fall` – Sadece düşüş (admin)

### Otomatik tarama

Scheduler her gün **09:00, 13:00, 17:00** (`.env`’deki `APP_TIMEZONE`) tam saatinde tarama yapar ve sonucu `TELEGRAM_CHAT_ID`’ye gönderir.

### Ana dosyalar

- `main.py` – Giriş, `run_scan_once`, scheduler hook
- `config.py` – Ayarlar, `.env` yükleme
- `bybit_client.py` – Bybit v5 REST (kline, funding, tickers)
- `scanner.py` – Tarama kriterleri, eşleşmeler
- `telegram_handler.py` – Komutlar, alarm mesajı formatı
- `scheduler.py` – 09:00 / 13:00 / 17:00 cron
- `ml/` – Özellik çıkarımı, veri seti, eğitim, tahmin

### Docker

```bash
docker build -t bybit-funding-bot .
docker run --env-file .env bybit-funding-bot
```
