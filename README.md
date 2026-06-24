## Bybit Funding Rate Scanner & Trading Bot

Bybit perpetual kontratlarında funding rate ve fiyat hareketlerini tarayan Telegram botu ve ML destekli otomatik işlem modülü.

### Mac'te Kurulum

1. **Gereksinimler:** Python 3.11+ (Homebrew ile: `brew install python`)

2. **Projeyi klonla:**

```bash
git clone https://github.com/CanYilmaz1907/HTC.git
cd HTC
```

3. **Sanal ortam oluştur ve bağımlılıkları yükle:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. **Ortam değişkenlerini ayarla:**

```bash
cp .env.example .env
```

`.env` dosyasını düzenle:

| Değişken | Açıklama |
|----------|----------|
| `TELEGRAM_BOT_TOKEN` | BotFather'dan aldığın bot token (`main.py` için zorunlu) |
| `TELEGRAM_CHAT_ID` | Bildirim alacağın chat ID |
| `ADMIN_USER_ID` | `/scan` gibi admin komutları için Telegram user ID |
| `BYBIT_API_KEY` | Bybit API anahtarı (gerçek işlem için) |
| `BYBIT_API_SECRET` | Bybit API secret |
| `BYBIT_BASE_URL` | Varsayılan: `https://api.bybit.com` (testnet için `https://api-testnet.bybit.com`) |

5. **Çalıştır:**

Telegram tarama botu:

```bash
python main.py
```

ML destekli otomatik işlem (gerçek hesap):

```bash
python BybitRealTradingExploit.py
```

Paper test (API anahtarı gerekmez): `BybitRealTradingExploit.py` içinde `SIMULATE_PAPER_1H = True` yap, sonra:

```bash
python BybitRealTradingExploit.py
```

### Docker ile Çalıştırma

```bash
docker build -t bybit-funding-bot .
docker run --env-file .env bybit-funding-bot
```

### Ana Bileşenler

- `config.py` — Ortak ayarlar ve `.env` yükleme
- `bybit_client.py` — Bybit v5 REST API istemcisi (async)
- `scanner.py` — Funding + fiyat taraması, SQLite
- `telegram_handler.py` — Telegram komutları (`/start`, `/status`, `/settings`, `/scan`)
- `scheduler.py` — APScheduler cron job'ları
- `main.py` — Telegram bot giriş noktası
- `BybitRealTradingExploit.py` — ML sinyalli otomatik işlem
- `ml/` — Eğitilmiş model (`model.joblib`, `scaler.joblib`)

### Notlar

- `.env` dosyası asla Git'e eklenmez; sadece `.env.example` şablon olarak paylaşılır.
- Mac'te `python` yerine `python3` kullanman önerilir.
- Gerçek işlem öncesi testnet veya paper mod ile dene.
