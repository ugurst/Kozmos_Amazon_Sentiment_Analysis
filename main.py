from warnings import filterwarnings
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import classification_report
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_excel("amazon.xlsx")
df.head()

# Veri seti belirli bir ürün grubuna ait yapılan yorumları, yorumbaşlığını, yıldız sayısını ve
# yapılan yorumu kaç kişinin faydalı bulduğunu belirten değişkenlerden oluşmaktadır.

#    Star  HelpFul                                          Title                                             Review
# 0     5        0                                    looks great                                      Happy with it
# 1     5        0  Pattern did not align between the two panels.  Good quality material however the panels are m...
# 2     5        0               Imagery is stretched. Still fun.  Product was fun for bedroom windows.<br />Imag...
# 3     5        0                 Que se ven elegantes muy finas   Lo unico que me gustaria es que sean un poco ...
# 4     5        0                             Wow great purchase  Great bang for the buck I can't believe the qu...

# Star: Ürüne verilen yıldız sayısı
# HelpFul: Yorumu faydalı bulan kişi sayısı
# Title: Yorum içeriğine verilen başlık - kısa yorum
# Review: Ürüne yapılan yorum


################################
# Normalizing Case Folding
###############################

df["Review"] = df["Review"].str.lower() \
    .str.replace(r"[^\w\s]", '', regex=True) \
    .str.replace('\d', '', regex=True)

df.Review.head()

# 0                                        happy with it
# 1    good quality material however the panels are m...
# 2    product was fun for bedroom windowsbr imagery ...
# 3     lo unico que me gustaria es que sean un poco ...
# 4    great bang for the buck i cant believe the qua...

#######################
# Stopwords
######################

sw = stopwords.words("english")
df["Review"] = df["Review"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# 0                                                happy
# 1      good quality material however panels mismatched
# 2    product fun bedroom windowsbr imagery bit stre...
# 3    lo unico que gustaria es que sean un poco mas ...
# 4    great bang buck cant believe quality material ...


#######################
# Rarewords / Custom Words
######################

temp_df = pd.Series(' '.join(df["Review"]).split()).value_counts()[-1000:]
temp_df.head()

# keen              1
# interpretation    1
# greatwould        1
# percect           1
# nother            1
# ...

df["Review"] = df["Review"].apply(lambda x: " ".join(x for x in x.split() if x not in temp_df))
df.head()

#######################
# Lemmatization
######################

df["Review"] = df["Review"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Görselleştirme

tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show(block=True)

# Wordcloud

text = " ".join(i for i in df.Review)

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show(block=True)

# Sentiment Analys
# Adım 1: Python içerisindeki NLTK paketinde tanımlanmış olan SentimentIntensityAnalyzer nesnesini oluşturunuz.
#
# Adım 2:  SentimentIntensityAnalyzer nesnesi ile polarite puanlarını inceleyiniz;
# a.	"Review" değişkeninin ilk 10 gözlemi için polarity_scores() hesaplayınız.
# b.	İncelenen ilk 10 gözlem için compund skorlarına göre filtreleyerek tekrar gözlemleyiniz.
# c.	10 gözlem için compound skorları 0'dan büyükse "pos" değilse "neg" şeklinde güncelleyiniz.
# d.	"Review" değişkenindeki tüm gözlemler için pos-neg atamasını yaparak yeni bir değişken olarak dataframe'e
# ekleyiniz.
#
# NOT: SentimentIntensityAnalyzer ile yorumları etiketleyerek, yorum sınıflandırma makine öğrenmesi modeli için bağımlı değiąken oluąturulmuş oldu.


sia = SentimentIntensityAnalyzer()

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

# 0   0.57
# 1   0.44
# 2   0.72
# 3   0.00
# 4   0.90
# 5   0.00
# 6   0.62
# 7   0.91
# 8   0.00
# 9   0.71

df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

# 0    pos
# 1    pos
# 2    pos
# 3    neg
# 4    pos
# 5    neg
# 6    pos
# 7    pos
# 8    neg
# 9    pos


df["Sentiment_Label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
df.head()

#    Star  HelpFul                                          Title                                             Review Sentiment_Label
# 0     5        0                                    looks great                                              happy             pos
# 1     5        0  Pattern did not align between the two panels.     good quality material however panel mismatched             pos
# 2     5        0               Imagery is stretched. Still fun.  product fun bedroom windowsbr imagery bit stre...             pos
# 3     5        0                 Que se ven elegantes muy finas  lo unico que gustaria e que sean un poco ma la...             neg
# 4     5        0                             Wow great purchase  great bang buck cant believe quality material ...             pos


df.groupby("Sentiment_Label")["Star"].mean()

train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["Sentiment_Label"],
                                                    random_state=42)

# Görev 4: Makine Öğrenmesine Hazırlık
# Adım 1: Bağımlı ve bağımsız değişkenlerimizi belirleyerek datayı train test olarak ayırınız.
#
# Adım 2: Makine öğrenmesi modeline verileri verebilmemiz için temsil şekillerini sayısala çevirmemiz gerekmekte;
# a.	TfidfVectorizer kullanarak bir nesne oluşturunuz.
# b.	Daha önce ayırmış olduğumuz train datamızı kullanarak oluşturduğumuz nesneye fit ediniz.
# c.	Oluşturmuş olduğumuz vektörü train ve test datalarına transform işlemini uygulayıp kaydediniz.


tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

# Görev 5:  Modelleme (Lojistik Regresyon)
# Adım 1: Lojistik regresyon modelini kurarak train dataları ile fit ediniz.
#
# Adım 2: Kurmuş olduğunuz model ile tahmin işlemleri gerçekleştiriniz;
# a.	Predict fonksiyonu ile test datasını tahmin ederek kaydediniz.
# b.	classification_report ile tahmin sonuçlarınızı raporlayıp gözlemleyiniz.
# c.	cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız.
#
# Adım 3: Veride bulunan yorumlardan ratgele seçerek modele sorulması;
# a.	sample fonksiyonu ile "Review" değişkeni içerisinden örneklem seçerek yeni bir değere atayınız.
# b.	Elde ettiğiniz örneklemi modelin tahmin edebilmesi için CountVectorizer ile vektörleştiriniz.
# c.	Vektörleştirdiğiniz örneklemi fit ve transform işlemlerini yaparak kaydediniz.
# d.	Kurmuş olduğunuz modele örneklemi vererek tahmin sonucunu kaydediniz.
# e.	Örneklemi ve tahmin sonucunu ekrana yazdırınız.
# Adım 1: Lojistik regresyon modelini kurarak train dataları ile fit ediniz.
#
# Adım 2: Kurmuş olduğunuz model ile tahmin işlemleri gerçekleştiriniz;
# a.	Predict fonksiyonu ile test datasını tahmin ederek kaydediniz.
# b.	classification_report ile tahmin sonuçlarınızı raporlayıp gözlemleyiniz.
# c.	cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız.
#
# Adım 3: Veride bulunan yorumlardan ratgele seçerek modele sorulması;
# a.	sample fonksiyonu ile "Review" değişkeni içerisinden örneklem seçerek yeni bir değere atayınız.
# b.	Elde ettiğiniz örneklemi modelin tahmin edebilmesi için CountVectorizer ile vektörleştiriniz.
# c.	Vektörleştirdiğiniz örneklemi fit ve transform işlemlerini yaparak kaydediniz.
# d.	Kurmuş olduğunuz modele örneklemi vererek tahmin sonucunu kaydediniz.
# e.	Örneklemi ve tahmin sonucunu ekrana yazdırınız.


log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)

# Tahmin

y_pred = log_model.predict(x_test_tf_idf_word)

print(classification_report(y_pred, test_y))

#               precision    recall  f1-score   support
#          neg       0.33      0.90      0.49        82
#          pos       0.99      0.89      0.94      1321
#     accuracy                           0.89      1403
#    macro avg       0.66      0.89      0.71      1403
# weighted avg       0.95      0.89      0.91      1403

cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean()

# 0.8546034570411795

random_review = pd.Series(df["Review"].sample(1).values)
new_comment = CountVectorizer().fit(train_x).transform(random_review)
pred = log_model.predict(new_comment)
print(f"Review: {random_review[0]} \n Prediction: {pred}")

# Görev 6:  Modelleme (Random Forest)
# Adım1:  Random Forest modeli ile tahmin sonuçlarının gözlenmesi;
# a.RandomForestClassifier modelini kurup fit ediniz.
# b.Cross validation fonksiyonunu kullanarak ortalama accuracy değerini hesaplayınız.


rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=-1).mean()

# 0.8923792577529233
