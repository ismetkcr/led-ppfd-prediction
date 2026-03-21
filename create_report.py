"""Generate LED_PPFD_Project_Report.docx using python-docx."""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
VIDEOS  = os.path.join(BASE, 'video_materials')
OUT     = os.path.join(BASE, 'LED_PPFD_Project_Report.docx')

# ── helpers ──────────────────────────────────────────────────────────────────

def add_page_number(doc):
    for section in doc.sections:
        footer = section.footer
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.clear()
        run = p.add_run("Sayfa ")
        def fld_char(typ):
            e = OxmlElement('w:fldChar'); e.set(qn('w:fldCharType'), typ); return e
        def instr(txt):
            e = OxmlElement('w:instrText'); e.text = txt
            e.set(qn('xml:space'), 'preserve'); return e
        run._r.append(fld_char('begin')); run._r.append(instr('PAGE')); run._r.append(fld_char('end'))
        run2 = p.add_run(" / ")
        run2._r.append(fld_char('begin')); run2._r.append(instr('NUMPAGES')); run2._r.append(fld_char('end'))
        for r in p.runs:
            r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x70, 0x70, 0x70)


def set_cell_bg(cell, hex_color):
    tc = cell._tc; tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color); tcPr.append(shd)


def h1(doc, text):
    p = doc.add_heading(text, level=1)
    if p.runs: p.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
    return p


def h2(doc, text):
    p = doc.add_heading(text, level=2)
    if p.runs: p.runs[0].font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)
    return p


def body(doc, text, bold=False, italic=False, size=11, color=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold; run.italic = italic; run.font.size = Pt(size)
    if color: run.font.color.rgb = color
    p.paragraph_format.space_after = Pt(4)
    return p


def bullet(doc, items, numbered=False):
    style = 'List Number' if numbered else 'List Bullet'
    for item in items:
        p = doc.add_paragraph(style=style)
        run = p.add_run(item); run.font.size = Pt(10.5)
        p.paragraph_format.space_after = Pt(2)


def insert_image(doc, path, width_in=6.0, caption=None, center=True):
    if not os.path.exists(path):
        print(f"  SKIP (not found): {path}"); return
    p = doc.add_paragraph()
    if center: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width_in))
    if caption:
        cp = doc.add_paragraph(caption)
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in cp.runs:
            r.font.size = Pt(9); r.font.italic = True
            r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
        cp.paragraph_format.space_after = Pt(8)


def insert_two_images(doc, path1, path2, cap1='', cap2='', width_each=2.9):
    """Put two images side by side in a 1-row 2-col table."""
    for path in (path1, path2):
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}"); return
    tbl = doc.add_table(rows=2, cols=2)
    tbl.style = 'Table Grid'
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for col_idx, (path, cap) in enumerate([(path1, cap1), (path2, cap2)]):
        cell = tbl.rows[0].cells[col_idx]
        p = cell.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(path, width=Inches(width_each))
        set_cell_bg(cell, 'FFFFFF')
        cap_cell = tbl.rows[1].cells[col_idx]
        cp = cap_cell.paragraphs[0]; cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cp.add_run(cap)
        run.font.size = Pt(8.5); run.italic = True
        run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
        set_cell_bg(cap_cell, 'FFFFFF')
    doc.add_paragraph()


def hr(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single'); bottom.set(qn('w:sz'), '4')
    bottom.set(qn('w:space'), '1'); bottom.set(qn('w:color'), 'BFBFBF')
    pBdr.append(bottom); pPr.append(pBdr)
    p.paragraph_format.space_after = Pt(6)


def simple_table(doc, headers, rows, col_widths_in, header_fill='2E75B6',
                 row_fills=None):
    """Generic bordered table with coloured header."""
    n_cols = len(headers)
    tbl = doc.add_table(rows=1 + len(rows), cols=n_cols)
    tbl.style = 'Table Grid'
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, (hdr, w) in enumerate(zip(headers, col_widths_in)):
        cell = tbl.rows[0].cells[j]
        cell.text = hdr
        set_cell_bg(cell, header_fill)
        dxa = str(int(w * 1440))
        tc = cell._tc; tcPr = tc.get_or_add_tcPr()
        tcW = OxmlElement('w:tcW'); tcW.set(qn('w:w'), dxa); tcW.set(qn('w:type'), 'dxa')
        tcPr.append(tcW)
        for para in cell.paragraphs:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.bold = True; run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    fills = row_fills or (['EDF2F9', 'F5F9FF'] * 20)
    for r_idx, row_data in enumerate(rows):
        fill = fills[r_idx % len(fills)]
        for c_idx, val in enumerate(row_data):
            cell = tbl.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val); set_cell_bg(cell, fill)
            for para in cell.paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs: run.font.size = Pt(10)
    doc.add_paragraph()


# ── Document ──────────────────────────────────────────────────────────────────

doc = Document()

for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

style = doc.styles['Normal']
style.font.name = 'Calibri'; style.font.size = Pt(11)

# ── TITLE PAGE ────────────────────────────────────────────────────────────────

doc.add_paragraph()
tp = doc.add_paragraph(); tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
tr = tp.add_run("LED Isik Dagilimi Tahmini")
tr.bold = True; tr.font.size = Pt(22)
tr.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D); tr.font.name = 'Calibri'

tp2 = doc.add_paragraph(); tp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
tr2 = tp2.add_run("Lineer Regresyon, YSA ve PINN ile Karsilastirmali Analiz")
tr2.bold = True; tr2.font.size = Pt(16)
tr2.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)

sp = doc.add_paragraph(); sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
sr = sp.add_run("Tarimsal Aydinlatma icin Makine Ogrenimi Tabanli PPFD Modelleme Sistemi")
sr.font.size = Pt(12); sr.italic = True
sr.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

doc.add_paragraph()

note_p = doc.add_paragraph(); note_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
note_r = note_p.add_run(
    "Veri Seti: Apogee Instruments Quantum PAR sensoru ile "
    "deneysel olarak tarafimdan toplanmistir."
)
note_r.italic = True; note_r.font.size = Pt(11)
note_r.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)

doc.add_paragraph(); hr(doc); doc.add_paragraph()

# ── SECTION 1: DENEYSEL KURULUM ───────────────────────────────────────────────

h1(doc, "1. Deneysel Kurulum ve Veri Toplama")
body(doc, (
    "Veri seti, gercek bir dikey tarim aydinlatma duzenegi uzerinde Apogee Instruments "
    "Quantum PAR sensoru kullanilarak tarafimdan deneysel olarak toplanmistir. "
    "Her yukseklikte 15 farkli olcum noktasinda tekrarli olcumler yapilmistir."
))

insert_two_images(
    doc,
    os.path.join(VIDEOS, 'sistem_uzak.jpeg'),
    os.path.join(VIDEOS, 'mor_kullan.jpeg'),
    cap1='Genel kurulum gorunumu',
    cap2='Quantum LED (mor spektrum) yakin cekim',
    width_each=2.8
)

insert_two_images(
    doc,
    os.path.join(VIDEOS, 'duzlem_kullan.jpeg'),
    os.path.join(VIDEOS, 'WhatsApp Image 2025-12-14 at 21.01.45.jpeg'),
    cap1='15 olcum noktasi duzlemi',
    cap2='PAR sensor ve olcum kurulumu',
    width_each=2.8
)

# ── SECTION 2: PROJE OZETI ────────────────────────────────────────────────────

h1(doc, "2. Proje Ozeti")
body(doc, (
    "Bu proje, LED aydinlatma sistemlerinden kaynaklanan PPFD (Photosynthetic Photon Flux Density) "
    "degerlerini farkli yuksekliklerde ve konumlarda tahmin etmek amaciyla gelistirilmistir. "
    "Lineer Regresyon, Yapay Sinir Agi (YSA) ve Fizik-Bilgili Sinir Agi (PINN) olmak uzere "
    "uc farkli yaklasim karsilastirmali olarak sunulmaktadir."
))
bullet(doc, [
    "Farkli LED tipleri icin PPFD dagilimini modellemek (Quantum, 12V, 54V)",
    "Uc modeli ayni veri seti uzerinde karsilastirarak guclu ve zayif yonlerini ortaya koymak",
    "Gercek tarimsal uygulamalar icin kullanilabilir bir tahmin araci gelistirmek",
    "Tkinter tabanli interaktif bir GUI ile kullanici dostu arayuz sunmak",
])

# ── SECTION 3: PROBLEM ────────────────────────────────────────────────────────

h1(doc, "3. Problem Tanimi")
body(doc, (
    "Tarimsal aydinlatmada LED panellerin kurulumu icin en kritik parametre, belirli bir "
    "yukseklik ve konumda olusan PPFD degeridir. Bitki buyumesi icin optimal PPFD araligi "
    "160-320 umol/m2/s, DLI hedefi ~11.5 mol/m2/gun'dur."
))
bullet(doc, [
    "Her konuma PAR sensoru yerlestirmek pahali ve zaman alicidir",
    "LED tipine, kurulum yuksekligine ve konuma gore dagilim buyuk olcude degisir",
    "Teorik formuller gercek dagilimi tam yakalayamamaktadir",
])
body(doc, "Cozum: Gercek olcumlerle egitilen ML modelleri.", bold=True)
bullet(doc, [
    "Lineer Regresyon (Ustel Bozunma) -- Fiziksel parametreleri bulma",
    "YSA (Yapay Sinir Agi) -- Veri guduml koordinat-PPFD eslemesi",
    "PINN -- Fizik denklemlerini sinir agina entegre etme",
], numbered=True)

# ── SECTION 4: VERI SETI ──────────────────────────────────────────────────────

h1(doc, "4. Veri Seti")
h2(doc, "4.1 LED Tipleri ve Ozellikleri")
simple_table(doc,
    ["LED Tipi", "LED Sayisi", "Uzunluk", "Maliyet"],
    [("Quantum LED", "120", "100 cm", "2.200 TL"),
     ("12V LED",     "144", "100 cm",   "550 TL"),
     ("54V LED",      "72", "100 cm",   "850 TL")],
    col_widths_in=[1.8, 1.2, 1.2, 1.2],
    header_fill='1F497D',
    row_fills=['D9E8F5', 'EBF3FB', 'D9E8F5']
)

h2(doc, "4.2 Olcum Noktalari ve Yukseklikler")
body(doc, "15 olcum noktasi: X [-40, +40] cm | Y [-30, +30] cm | Merkez: (0, 0)")
body(doc, "3 nominal yukseklikte olcum yapilmistir:")
simple_table(doc,
    ["Nominal Yukseklik", "Gercek Yukseklik", "Offset"],
    [("13 cm", "13 cm", "0 cm"),
     ("27 cm", "31 cm", "+4 cm"),
     ("35 cm", "39 cm", "+4 cm")],
    col_widths_in=[1.8, 1.8, 1.8],
    header_fill='2E75B6'
)
body(doc, "PPFD donusumu: Ham olcum x 0.8 = gercek PPFD (aralik: 4 - 326.4 umol/m2/s)", italic=True)

# ── SECTION 5: LINEER REGRESYON ───────────────────────────────────────────────

h1(doc, "5. Model 1 -- Lineer Regresyon (Ustel Bozunma)")
h2(doc, "5.1 Fiziksel Model")
body(doc, "PPFD(x,y,h) = theta1 x SUM_i exp(-alpha x R_i)", bold=True)
body(doc, "R_i = sqrt( (x-x_i)^2 + (y-y_i)^2 + h^2 )")
bullet(doc, [
    "theta1: PPFD amplitudu (LED basina isik cikisi)",
    "alpha: Ustel bozunma katsayisi",
    "R_i: i. LED ile olcum noktasi arasindaki 3D Oklid mesafesi",
])
body(doc, "theta1 analitik kapal form: theta1_opt = (S . PPFD) / (S . S)", italic=True)

h2(doc, "5.2 Optimizasyon Yontemleri")
bullet(doc, [
    "Grid Search: [0.001, 1.0] araliginda 500 nokta",
    "Golden Section Search: Scipy minimize_scalar, tolerans 1e-12",
])

h2(doc, "5.3 Sonuclar")
simple_table(doc,
    ["Yontem", "alpha", "theta1", "R2", "RMSE (umol/m2/s)"],
    [("Grid Search",      "0.1732", "106.10", "0.784", "38.62"),
     ("Golden Section",   "0.1734", "106.56", "0.784", "38.62")],
    col_widths_in=[1.5, 1.0, 1.0, 0.8, 1.8]
)

insert_image(doc, os.path.join(RESULTS, 'linear_regression_results.png'),
             width_in=6.0,
             caption="Sekil 1 -- Lineer Regresyon: Grid Search, Golden Section ve SSE vs Alpha egrisi")

h2(doc, "5.4 Degerlendirme")
body(doc, "Avantajlar:", bold=True)
bullet(doc, [
    "Yorumlanabilir fiziksel parametreler (alpha, theta1)",
    "Hizli hesaplama, matematiksel seffaflik",
])
body(doc, "Sinirlamalar:", bold=True)
bullet(doc, [
    "Tek alpha ile tum yukseklikleri modellemek yetersiz (R2=0.78)",
    "Basit ustel model gercek isik dagilimini tam yakalayamiyor",
])

# ── SECTION 6: YSA ────────────────────────────────────────────────────────────

h1(doc, "6. Model 2 -- Yapay Sinir Agi (YSA / Feed-Forward ANN)")
h2(doc, "6.1 Mimari")
body(doc, "Giris [x, y, h]  ->  64 (ReLU)  ->  32 (ReLU)  ->  16 (ReLU)  ->  PPFD", bold=True)
bullet(doc, [
    "Giris Kati: 3 noron [x, y, h]",
    "Gizli Kat 1: 64 noron, ReLU aktivasyonu",
    "Gizli Kat 2: 32 noron, ReLU aktivasyonu",
    "Gizli Kat 3: 16 noron, ReLU aktivasyonu",
    "Cikis Kati: 1 noron (lineer, PPFD tahmini)",
])

h2(doc, "6.2 Egitim ve Veri Bolunmesi")
simple_table(doc,
    ["Kume", "Yukseklikler", "Nokta Sayisi", "Amac"],
    [("Egitim", "13 cm + 39 cm", "30", "Model ogrenimi"),
     ("Test",   "31 cm",         "15", "Interpolasyon testi")],
    col_widths_in=[1.2, 1.6, 1.2, 2.5]
)
body(doc, "Optimizer: Adam  |  Loss: MSE  |  Epoch: 1000  |  Batch: 4  |  Normalizasyon: StandardScaler")

h2(doc, "6.3 Sonuclar")
simple_table(doc,
    ["Kume", "R2", "RMSE (umol/m2/s)"],
    [("Egitim (13+39 cm)",   "0.9999", "0.67"),
     ("Test (31 cm)",        "0.5808", "21.55")],
    col_widths_in=[2.4, 1.2, 2.0]
)

insert_image(doc, os.path.join(RESULTS, 'ann_results.png'),
             width_in=6.0,
             caption="Sekil 2 -- YSA Performansi: Egitim (13+39 cm) ve Test (31 cm interpolasyon)")

h2(doc, "6.4 Degerlendirme")
body(doc, "Avantajlar:", bold=True)
bullet(doc, [
    "Egitim verisinde mukemmel uyum (R2~1.0)",
    "Herhangi koordinat ve LED tipi icin kullanilabilir",
])
body(doc, "Sinirlamalar:", bold=True)
bullet(doc, [
    "Overfitting: egitim/test R2 farki buyuk (0.9999 vs 0.58)",
    "Az veri ile genelleme zorluyor",
    "Fiziksel yorum yok (kara kutu modeli)",
])

# ── SECTION 7: PINN ───────────────────────────────────────────────────────────

h1(doc, "7. Model 3 -- Fizik-Bilgili Sinir Agi (PINN)")
h2(doc, "7.1 Neden PINN Gerekli?")
simple_table(doc,
    ["Sorun", "PINN Cozumu"],
    [("Lineer Reg --> dusuk R2 (0.78)", "FFNN ile yukseklige bagli A(h) ogrenilir"),
     ("YSA --> overfitting (test R2=0.58)", "Fizik denklemi kisit olarak ekleniyor"),
     ("Az veri", "Fiziksel denklem eksik veriyi telafi eder")],
    col_widths_in=[2.8, 3.6],
    header_fill='1F497D'
)

h2(doc, "7.2 Model Mimarisi")
body(doc, "h (yukseklik)  ->  FFNN [64->32->8->1]  ->  A(h)", bold=True)
body(doc, "PPFD = A(h) x SUM_i [ h / R_i^3 x exp(-alpha x R_i) ]", bold=True)
bullet(doc, [
    "A(h): Yukseklige bagli amplitud -- FFNN tarafindan ogrenilir",
    "alpha: Global bozunma katsayisi -- ogrenilen parametre",
    "LED konumlari: Modele DISARIDAN verilir -- agirliklara gomulu degil!",
    "Quantum LED icin alpha = 0.0428 cm-1 bulunmustur",
])

h2(doc, "7.3 Temel Ozellik: LED Konumlarinin Dissal Olmasi")
body(doc, (
    "PINN'in en guclu yonu, LED konumlarinin model agirliklarina gomulu olmamasi ve "
    "dis parametre olarak saglanmasidir. Bu sayede model;"
))
bullet(doc, [
    "Farkli LED orientasyonlari (yatay/dikey) ile kullanilabilir",
    "Coklu lamba kurulumlari (superpozisyon prensibi) desteklenebilir",
    "Farkli LED sayisina sahip sistemlere uygulanabilir",
    "Agirliklar degistirilmeden farkli konfigurasyonlar test edilebilir",
])

h2(doc, "7.4 Egitim Parametreleri")
body(doc, "Learning rate: 0.0005  |  Epoch: 5000  |  Batch: 4  |  Egitim: 13+27+35 cm")

h2(doc, "7.5 Sonuclar -- Quantum LED (Tum Yuksekliklerde)")
simple_table(doc,
    ["Yukseklik", "Gercek Yukseklik", "R2", "RMSE (umol/m2/s)"],
    [("13 cm", "13 cm", "0.9826", "15.05"),
     ("27 cm", "31 cm", "0.9630",  "6.40"),
     ("35 cm", "39 cm", "0.8900",  "7.69")],
    col_widths_in=[1.4, 1.6, 1.0, 2.4]
)

insert_image(doc, os.path.join(RESULTS, 'pinn_measured_vs_predicted.png'),
             width_in=6.0,
             caption="Sekil 3 -- PINN: Olculen vs Tahmin edilen PPFD (tum 3 yukseklik)")

insert_image(doc, os.path.join(RESULTS, 'pinn_scatter.png'),
             width_in=6.0,
             caption="Sekil 4 -- PINN Scatter: Tahmin edilen vs Olculen PPFD")

h2(doc, "7.6 Kaydedilen Modeller (Tum LED Tipleri)")
simple_table(doc,
    ["LED Tipi", "R2 (Egitim)", "RMSE (umol/m2/s)", "LED Sayisi"],
    [("Quantum", "0.9781", "10.44", "120"),
     ("12V",     "0.9546",  "4.01", "144"),
     ("54V",     "0.9793",  "6.53",  "72")],
    col_widths_in=[1.5, 1.5, 2.0, 1.4]
)

# ── SECTION 8: KARSILASTIRMA ──────────────────────────────────────────────────

h1(doc, "8. Model Karsilastirmasi")
body(doc, "Quantum LED uzerinde uc modelin performans karsilastirmasi:")

simple_table(doc,
    ["Model", "R2 (Egitim)", "R2 (Test)", "RMSE", "Fizik", "Yorumlanabilirlik"],
    [("Lineer Regresyon", "0.784",  "0.784", "38.62", "Evet", "Yuksek"),
     ("YSA (ANN)",       "0.9999", "0.580",  "0.67", "Hayir", "Dusuk (kara kutu)"),
     ("PINN",            "0.978",  "--",    "10.44", "Evet",  "Yuksek")],
    col_widths_in=[1.5, 1.1, 1.1, 1.0, 0.8, 1.9],
    header_fill='1F497D',
    row_fills=['EDF2F9', 'F5F9FF', 'EDF2F9']
)

body(doc, "Sonuc:", bold=True)
bullet(doc, [
    "En Iyi Genel Performans: PINN (fizik + veri uyumu dengesi)",
    "En Yorumlanabilir Model: Lineer Regresyon",
    "En Yuksek Egitim Uyumu: YSA (ancak overfitting)",
    "En Genellestirilebilir: PINN (farkli LED konfigurasyonlarina uyarlanabilir)",
])

# ── SECTION 9: GUI ────────────────────────────────────────────────────────────

h1(doc, "9. Kullanici Arayuzu (GUI)")
body(doc, (
    "Tkinter tabanli interaktif masaustu uygulamasi — 5 sekme. "
    "Veri yukleme, model egitimi, sonuc gorsellestiirme ve interaktif tahmin "
    "hesaplayici fonksiyonlarini icerir."
))

simple_table(doc,
    ["Sekme", "Icerik"],
    [("Gorsellestiirme",    "PPFD isi haritalari, tum LED/yukseklik kombinasyonlari"),
     ("Lineer Regresyon",  "Parametre tahmini, SSE grafigi, interaktif hesaplayici"),
     ("YSA Tahmini",       "Yapilandiriabilir mimari (1-5 kat, 4-256 noron), gercek zamanli egitim"),
     ("PINN Tahmini",      "A(h) ve alpha grafikleri, model kaydet/yukle, PPFD tahmin araci"),
     ("Ozel LED Duzeni",   "Ozel konfigürasyon tasarimi, coklu lamba simulasyonu")],
    col_widths_in=[1.8, 4.6],
    header_fill='2E75B6'
)
body(doc, "Calistirmak icin: python new_led_visualization_gui.py", italic=True)

h2(doc, "9.1 Ozel LED Duzeni Ekran Goruntuleri")
SHOTS = os.path.join(BASE, 'screenshots')

_s3layout = os.path.join(SHOTS, 'custom_layout_3leds.png')
_s3heat   = os.path.join(SHOTS, 'custom_layout_heat_map_3_leds.png')
_s5layout = os.path.join(SHOTS, 'custom_layout_5_led.png')
_s5heat   = os.path.join(SHOTS, 'custom_layout_5_led_heat_map.png')
_s3stats  = os.path.join(SHOTS, 'custom_layout_3_led_statistics.png')

if os.path.exists(_s3layout) and os.path.exists(_s3heat):
    insert_two_images(doc, _s3layout, _s3heat,
                      cap1='3x 54V LED kurulum duzeni',
                      cap2='3 LED PPFD isi haritasi')
if os.path.exists(_s5layout) and os.path.exists(_s5heat):
    insert_two_images(doc, _s5layout, _s5heat,
                      cap1='5 LED karisik kurulum (Quantum + 54V)',
                      cap2='5 LED PPFD dagilimi')
if os.path.exists(_s3stats):
    insert_image(doc, _s3stats, width_in=5.5,
                 caption='Istatistik paneli — 3x 54V LED kurulumu')

# ── SECTION 10: TEKNOLOJILER ──────────────────────────────────────────────────

h1(doc, "10. Kullanilan Teknolojiler")
simple_table(doc,
    ["Teknoloji", "Kullanim"],
    [("Python 3.12",          "Ana programlama dili"),
     ("TensorFlow / Keras",   "YSA ve PINN modelleri"),
     ("NumPy",                "Matris hesaplari, mesafe matrisi"),
     ("Pandas",               "CSV veri isleme"),
     ("Matplotlib",           "Tum grafikler ve isi haritalari"),
     ("Scikit-learn",         "StandardScaler normalizasyon"),
     ("SciPy",                "Golden Section Search optimizasyonu"),
     ("Tkinter",              "GUI arayuzu"),
     ("h5py",                 "Model agirliklarini okuma/yazma")],
    col_widths_in=[2.2, 4.2]
)

# ── SECTION 11: SONUC ─────────────────────────────────────────────────────────

h1(doc, "11. Sonuc ve Degerlendirme")
body(doc, (
    "Bu proje, LED aydinlatma sistemlerinde PPFD tahmini icin uc farkli yaklasimin "
    "kapsamli bir karsilastirmasini sunmaktadir. Deneysel olarak Quantum PAR sensoru "
    "ile toplanan gercek veri uzerinde test edilen PINN yaklasimi, fiziksel kisitlamalar "
    "ve veri uyumunu dengelemi, en iyi genellestirme performansini gostermistir."
))
body(doc, (
    "PINN'in 'LED konumlarinin harici parametre olmasi' ozelligi modeli uretim ortamlarinda "
    "yeniden kullanilabilir kilmaktadir. GUI araciligiyla kullanici dostu arayuz sunmasi "
    "bu projeyi pratik bir uygulama haline getirmektedir."
))

hr(doc)
body(doc, "CV portfoyu icin hazirlandi -- 2025", italic=True, size=9)

add_page_number(doc)
doc.save(OUT)
print(f"Saved: {OUT}")
