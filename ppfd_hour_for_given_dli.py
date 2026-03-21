import numpy as np
import matplotlib.pyplot as plt

# Parametreler
DLI_optimum = 11.5  # mol/m²/day
PPFD_min = 50
PPFD_max = 500
nokta_sayisi = 20

# PPFD değerleri (grafik için)
PPFD = np.linspace(PPFD_min, PPFD_max, nokta_sayisi)
Fotoperiod = (DLI_optimum * 277.78) / PPFD

# Tablo için değerler
PPFD_tablo = [50, 100, 150, 160, 180, 200, 220, 250, 300, 400, 500]
Fotoperiod_tablo = [(DLI_optimum * 277.78) / p for p in PPFD_tablo]

# Durum kontrolü
def durum_belirle(foto):
    if foto > 24:
        return "❌ >24h"
    elif foto > 20:
        return "⚠️ >20h"
    elif foto < 8:
        return "⚠️ <8h"
    else:
        return "✅ İdeal"

durumlar = [durum_belirle(f) for f in Fotoperiod_tablo]

# Figure oluştur
fig = plt.figure(figsize=(16, 7))

# Sol taraf: Grafik
ax1 = plt.subplot(1, 2, 1)

# Pratik aralık vurgusu (sadece yeşil bölge)
pratik_min_ppfd = DLI_optimum * 277.78 / 20
pratik_max_ppfd = DLI_optimum * 277.78 / 8
ax1.axvspan(pratik_min_ppfd, pratik_max_ppfd, alpha=0.15, color='green', label='Pratik PPFD Aralığı')

# Ana eğri
ax1.plot(PPFD, Fotoperiod, 'b-', linewidth=2.5, label=f'DLI = {DLI_optimum} mol/m²/day')
ax1.scatter(PPFD, Fotoperiod, color='red', s=50, zorder=5)

# Sınır çizgileri
ax1.axhline(y=24, color='gray', linestyle='--', linewidth=1.5, label='24 saat sınırı', alpha=0.7)
ax1.axhline(y=20, color='orange', linestyle='--', linewidth=1.5, label='20 saat (maks. önerilen)', alpha=0.7)
ax1.axhline(y=8, color='green', linestyle='--', linewidth=1.5, label='8 saat (min. önerilen)', alpha=0.7)

# Etiketler
ax1.set_xlabel('PPFD (µmol m⁻² s⁻¹)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Fotoperiod (saat)', fontsize=13, fontweight='bold')
ax1.set_title(f'PPFD - Fotoperiod İlişkisi\n(DLI = {DLI_optimum} mol/m²/day)', 
              fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax1.set_xlim(0, 550)
ax1.set_ylim(0, 30)

# Sağ taraf: Tablo
ax2 = plt.subplot(1, 2, 2)
ax2.axis('off')

# Tablo başlığı
ax2.text(0.5, 0.95, f'Optimum DLI: {DLI_optimum} mol/m²/day', 
         ha='center', va='top', fontsize=14, fontweight='bold',
         transform=ax2.transAxes)

# Tablo verisi
tablo_data = []
for i, (ppfd, foto, durum) in enumerate(zip(PPFD_tablo, Fotoperiod_tablo, durumlar)):
    tablo_data.append([f'{ppfd}', f'{foto:.1f}', durum])

# Tablo oluştur
table = ax2.table(cellText=tablo_data,
                  colLabels=['PPFD\n(µmol/m²/s)', 'Fotoperiod\n(saat)', 'Durum'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0.05, 0.15, 0.9, 0.75])

# Tablo stillendirme
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# Başlık satırı renklendirme
for i in range(3):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Satır renklendirme (sadece durum bazlı)
for i, ppfd in enumerate(PPFD_tablo):
    row = i + 1
    durum = durumlar[i]
    
    # Durum bazlı renklendirme
    if '❌' in durum:
        color = '#FFE6E6'  # Açık kırmızı
    elif '⚠️' in durum and '<8h' in durum:
        color = '#FFF4E6'  # Açık turuncu
    elif '⚠️' in durum:
        color = '#FFF4E6'  # Açık turuncu
    else:  # ✅
        color = '#E6F4EA'  # Açık yeşil
    
    for col in range(3):
        table[(row, col)].set_facecolor(color)

plt.tight_layout()
plt.show()

# Konsol çıktısı
print(f"\n{'='*70}")
print(f"Optimum DLI: {DLI_optimum} mol/m²/day")
print(f"{'='*70}")
print(f"{'PPFD':<15} {'Fotoperiod':<15} {'Durum':<25}")
print(f"{'-'*70}")
for ppfd, foto, durum in zip(PPFD_tablo, Fotoperiod_tablo, durumlar):
    print(f"{ppfd:<15} {foto:<15.1f} {durum:<25}")
print(f"{'='*70}")
print(f"\n📊 Pratik PPFD Aralığı: {pratik_min_ppfd:.0f} - {pratik_max_ppfd:.0f} µmol/m²/s (8-20 saat arası)")
print(f"{'='*70}\n")

