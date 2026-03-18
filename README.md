## Bybit Funding Rate Scanner Telegram Bot

Bu proje, Bybit perpetual kontratlarında funding rate ve kısa vadeli fiyat hareketlerini tarayarak
belirli kriterlere uyan coinleri Telegram üzerinden bildiren async bir bottur.

### Kurulum

1. Gerekli paketleri yükle:

```bash
pip install -r requirements.txt
```

2. `.env` dosyasını düzenle:

- `TELEGRAM_BOT_TOKEN`: BotFather'dan aldığın bot token
- `TELEGRAM_CHAT_ID`: Bildirim almak istediğin chat ID (kendi DM'in de olabilir)
- `ADMIN_USER_ID`: `/scan` gibi admin komutlarını kullanacak Telegram user ID
- `BYBIT_BASE_URL`: Test için `https://api-testnet.bybit.com` (varsayılan)

Diğer ayarlar isteğe bağlıdır.

3. Botu çalıştır:

```bash
python main.py
```

### Docker ile Çalıştırma

```bash
docker build -t bybit-funding-bot .
docker run --env-file .env bybit-funding-bot
```

### Ana Bileşenler

- `config.py`: Ortak ayarlar ve .env yükleme
- `bybit_client.py`: Bybit v5 REST API istemcisi (async, `aiohttp`)
- `scanner.py`: 09:00 fiyat kaydı, 09:05 funding + fiyat taraması, SQLite (`aiosqlite`)
- `telegram_handler.py`: Telegram komutları (/start, /status, /settings, /scan)
- `scheduler.py`: APScheduler ile 09:00 ve 09:05 cron job'ları
- `main.py`: Uygulama girişi, tüm bileşenleri başlatır

