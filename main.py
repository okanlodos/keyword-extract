# -*- coding:utf-8 -*-
from keyword_tfidf import tfidf
from textblob import TextBlob as tb


doc1 = u"""
Bu yazının amacı, literatürde metin madenciliği (text mining) veya metin veri madenciliği (text data mining) kavramını açıklamaktır.

En basit anlamda, metin madenciliği çalışmaları metni veri kaynağı olarak kabul eden veri madenciliği (data mining) çalışmasıdır diğer bir tanımla metin üzerinden yapısallaştırılmış (structured) veri elde etmeyi amaçlar. Örneğin metinlerin sınıflandırılması, bölütlenmesi (clustering), metinlerden konu çıkarılması (concept/entity extraction), sınıf taneciklerinin üretilmesi (production of granular taxonomy), duygusal analiz (sentimental analysis), metin özetleme (document summarization), varlık ilişki modellemesi (entity relationship modelling) gibi çalışmaları hedefler.

Yukarıdaki hedeflere ulaşılması için metin madenciliği çalışmaları kapsamında enformasyon getirimi (information retrieval), hece analizi (lexical analysis), kelime frekans dağılımı (Word requency distribution), örüntü tanıma (pattern recognition), etiketleme (tagging), enformasyon çıkarımı (information extraction), veri madenciliği (data mining) ve hatta görselleştirme (visualization) gibi yöntemleri kullanmaktadır.

Metin madenciliği çalışmaları, metin kaynaklı literatürdeki diğer bir çalışma alanı olan doğal dil işleme (natural language processing, NLP) çalışmaları ile çoğu zaman beraber yol yürümektedir. Doğal dil işleme çalışmaları daha çok yapay zeka altındaki dil bilim bilgisine dayalı çalışmalarını kapsamaktadır. Metin madenciliği çalışmaları ise daha çok istatistiksel olarak metin üzerinden sonuçlara ulaşmayı hedefler. Metin madenciliği çalışmaları sırasında çoğu zaman doğal dil işleme kullanılarak özellik çıkarımı da yapılmaktadır.

Genel olarak klasik bir metin madenciliği çalışmasını aşağıdaki şekilde özetleyebiliriz.

Metin_madenciligi_text_mining

Yukarıdaki şekilde de görüldüğü üzere, bir metin veri tabanından alınan veriler öncelikle bir özellik çıkarımına tabi tutulur. Ardından çıkarılan özellikler üzerinde bir makine öğrenmesi algoritması çalışır (sınıflandırma (classification), bölütleme (clustering), tahmin (prediction) v.b.) ve neticede yapılandırılmış veri (structured data) elde edilir.

Buradaki makine öğrenmesi aşaması genelde kullanılmakla birlikte, metin madenciliği için şart olmayan bir aşamadır. Bazı durumlarda, doğrudan çıkarılan özellik aranan yapılandırılmış veri olabilmektedir. Bazı durumlarda ise makine öğrenmesi adımı yerine, istatistiksel bazı farklı yöntemler kullanılabilir.

Metin kaynakları, genelde doğal dilde yazılmış kaynaklardır. Yani bir gazetedeki köşe yazıları, bir kitap, bir makale olabilir. Hatta internet üzerindeki web siteleri bile metin kaynağı olarak görülebilir (bu konu daha özel olarak web madenciliği (web mining) olarak da adlandırılmaktadır). Bu yazıların, metin madenciliği açısından önemli bir de üst bilgileri olması söz konusudur. Örneğin yazının tarihi, yazının yayınlandığı web sitesi, yazar bilgisi gibi, yazının içerisinde yer almayan ancak yazı ile ilgili metin madenciliğinde kullanılabilecek önemli üst bilgiler (meta data) bulunabilir.

Özellik çıkarımı (feature extraction) aşamasında, metinlerin doğrudan içeriğinden veya üst bilgilerinden yararlanılarak istenilen özellikler çıkarılabilir ve çıkarılan özellikler üzerinde işlem yapılabilir.

Örnek Metin Madenciliği uygulaması:

Örneğin elimizde 100 adet yazı olsun. Bu yazıları yazan yazarları biliyor olalım (diyelim ki 5 farklı yazarın 20’şer adet yazısı olsun). Yeni gelen 101. Yazının bu 5 yazardan hangisine ait olduğunu bulmak, klasik bir metin madenciliği uygulamasıdır ve literatürde yazar tanıma (author recognition) olarak da geçer.

Burada örnek olarak metinlerdeki kelime kullanma sıklıklarını özellik çıkarımı için kullanmak isteyelim. Yani yazarlarımızı kullandıkları kelime sıklıklarından tanıyabileceğimizi düşünüyoruz (author attribution). Her metinde ve dolayısıyla her yazar için hangi kelimeyi ne sıklıkla kullandığı bilgisi bizim özellik çıkarımı aşamamız oluyor.

Ardından kullanılan kelime sıklıklarını örnek olarak makine öğrenme algoritması olan KNN algoritmasına veriyoruz ve diyelim ki yazarını tanımak istediğimiz 101. Yazı için her kelime için en çok kullanan yazarları listeliyoruz. Neticede bize bir olası yazarlar listesi çıkıyor ve biz de en yüksek ihtimalle hangi yazarın bu yazıyı yazmış olabileceğini söylüyoruz. Bu çıkan sonuç aslında 101. Yazı için anlamlı ve yapılandırılmış bir sonuç olarak kabul edilebilir.

Metin madenciliğinin çalışma alanları:

Metin madenciliği sırasında genelde aşağıdaki problemlerle ilgilenilir (bunlarla sınırlı değildir).

Enformasyon Getirimi (Information Retrieval): Bu aşama ilgilenilen külliyet (derlem, corpus) hakkında ön bilginin toplandığı aşamadır. Örneğin metin madenciliği web üzerindeki veri kaynakları üzerinde yapılacaksa web sayfaları, adresleri veya dosya sistemi üzerindeyse dosyaların tarihleri, kullanıcı bilgileri, dosya isimleri, dizin bilgileri gibi bilgilerin toplandığı aşamadır.

Doğal dil işleme aşaması (natural language processing): Bu aşama bütün metin madenciliği aşamalarında kullanılmasa bile genelde özellik çıkarımı ve metinden bazı anlamsal bilgilerin elde edilmesinde sıklıkla başvurulan aşamadır. Örneğin, konuşma parçalarının etiketlenmesi (part of speech tagging) veya cümlebilimsel parçalama (syntactic parsing) veya diğer dilbilimsel işlemler doğal dil işleme aşamasında yapılır.

Adlandırılmış varlık tanıma (named entity recognition): Genellikle metin işleme aşamasında istatistiksel bazı özelliklerin çıkarılması için kullanılır. Örneğin, metnin içerisindeki kişi isimleri, yer isimleri, semboller, kısaltmalar v.s. bu yöntemle bulunur. Metin madenciliği çalışmalarının her zaman temiz metinlerde yapılmadığını hatırlatmakta yarar vardır. Örneğin facebook, twitter mesajları, telefonlardan yollanan SMS mesajları gibi mesajların çoğunda yazım hataları hatta kısaltmalar kullanılmaktadır. Metin madenciliği bu ihtimallerin de göz önünde tutulması gereken çalışmalardır. Örneğin ‘’osmanbey’’ kelimesi, istanbulda bir semt ismi olabileceği gibi bir kişi ismi de olabilir. Adlandırılmış varlık tanıma çalışmalarında, hedeflenen kelime gruplarının metin içerisinden çıkarılması, sayılması, yoğunluğunun bulunması, etiketlenmesi gibi işlemler yapılabilir.

Örüntüsü tanımlı varlıkların bulunması (pattern identified entities): Bazı durumlarda, metnin içerisinden özel bazı bilgilerin metin madenciliğine konu olması mümkündür. Örneğin e-posta adresleri, telefon numaraları, adresler, tarihler gibi bazı bilgileri özel olarak almak isteyebiliriz. Genelde bu durumlarda düzenli ifadeler (regular expressions) veya içerik bağımsız gramerler (context free grammers) tanımlanarak metin üzerinde çalıştırılır[1].

Eş Atıf (Coreference): Bir varlığa işaret eden (atıf eden) isim kelime gruplarını ve diğer terimlerin bulunması/ayrılmasını hedefler.

İlişki, kural, olay çıkarımları: Çeşitli amaçlarla metnin içerisinden bazı bilgilerin çıkarılması istenebilir. Örneğin doktora çalışmam sırasında, verilen bir metnin içerisindeki olayları çıkararak sıralamak (event ordering) üzerine çalışmış, Türkçedeki fiil yapılarını, olay belirten kelime gruplarını, zaman kalıplarını ve bütün bu kelime grupları arasındaki olası ilişkileri gösteren özel bir matematik tasarlamıştım[2].

Duygu analizi (sentimental Analysis) : Metinlerde geçen duygusal ifadelerin çıkarılmasını amaçlar. En sık kullanılanı duygusal kutupsallıktır (sentimental polarity). Buna göre bir konu hakkında geçen mesajların veya yazıların olumlu veya olumsuz olmasına göre iki sınıfa ayrılması hedeflenir. Ancak duygu analizi bunun dışında, metinlerdeki ruh hali, kanaat ve daha karmaşık duyguların çıkarılması üzerinde de çalışmaktadır.
"""

doc2 = u"""
Bu yazının amacı, bir makine öğrenmesi (machine learning) ve veri madenciliği (data mining) aracı olan ve iş zekası (business intelligence) gibi farklı alanlarda kullanımı olan WEKA aracının üzerinde yapılan eğitim modellerinin nasıl kaydedilip, farklı test kümeleri üzerine nasıl uygulandığını anlatmaktır.

WEKA arcını kullanan kişilerin yaşadığı bir durum, WEKA’nın en sık kullanılan ve en kolay ekranı olan Explorer ekranındaki eğitim (train) ve test kümelerinin aynı anda veriliyor olması ve farklı test kümelerinin verilemiyor olmasıdır. Oysaki WEKA’nın en kuvvetli özelliklerinden birisi olan, JAVA ile tam uyum sağlayan yapısı, yazılan kodlarda aynı eğitim modelinin tekrar tekrar eğitilmeden kullanılmasını mümkün kılmaktadır.

Yani modelimizi bir kere eğitip sonra çok farklı testler için kullanabiliriz. Mesela, bir güvenlik firmasının kamera üzerinden yüz tanıma programını yazdığımızı düşünelim. Elimizdeki yüzlerce personelin yüzlerini sisteme tanıtıp modelimizi eğittikten sonra bu eğitilmiş modeli, her yeni gelen kişi için kullanmak gerekir. Aksi halde her yeni kişi için yeniden yüzlerce kişinin yüzünü sistemde eğitmemiz ve bunun için zaman ve donanım kaynaklarını israf etmemiz gerekir ki pek de mantıklı bir yol olduğu söylenemez.

Deneme aşamasında, WEKA’yı konsoldan çağırmak iyi bir çözümdür.

java weka.classifiers.trees.J48 -t egitim.arff -d egitilmis.model

Yukarıdaki komut ile J48 karar ağacını, verdiğimiz egitim.arff dosyası üzerinden eğitiyoruz ve eğitilmiş modeli egitilmis.model isimli dosyaya kaydediyoruz.

java weka.classifiers.trees.J48 -l egitilmis.model -T test.arff

Yukarıdaki komutu ise istediğimiz kadar, istediğimiz farklı dosyalar ile çağırabiliriz. Burada ise bir önceki adımda kaydettiğimiz egitilmis.model dosyasını kullanarak test.arff dosyasındaki verilerin sınıflandırılmasını istiyoruz. egitilmis.model dosyasını oluşturduktan sonra ikinci adımı istediğiniz kadar farklı dosyaya uygulayabilirsiniz.

Yukarıdaki, ikinci komutu, istediğimiz kadar farklı test dosyası üzerine uygulamamız ve sonuçlarını almamız mümkündür. Ancak yukarıdaki yaklaşımda bir problem, yapmış olduğumuz eğitim ve test işlemlerinin yazılım ile entegre olamamasıdır.

Bunun için biraz java kodlamaya girip yukarıdaki işlemin aynısını JAVA kodumuza entegre etmeye çalışalım. Bu yaklaşımda, java’nın bir özelliği olan nesne serileme (object serialization) özelliğini kullanacağız.

Ancak öncelikle nesne serileme işlemini kısaca hatıraltalım. Nesne serileme, basitçe bir nesnenin bütün bilgilerinin dizgiye (string) çevrilmesidir. Bu çevirim ne işe yarar derseniz, hafızada (RAM) yaşayan nesne bir kere dizgiye (string) çevrildikten sonra artık dosyaya yazılabilir veya ağ üzerinden farklı bir kaynağa yollanabilir. Biz de bu özelliği, java hafızada bir nesne olarak tuttuğumuz eğitilmiş model üzerine uygulayacağız ve bu sayede nesnenin içindeki eğitilmiş modeli her seferinde tekrar tekrar kullanabileceğiz.

// Modelin tanimi
 Classifier cls = new J48();

 // Egitim
 Instances inst = new Instances(
                    new BufferedReader(
                      new FileReader("egitimkumesi.arff")));
 inst.setClassIndex(inst.numAttributes() - 1);
 cls.buildClassifier(inst);

 // modelin serilenmesi
 ObjectOutputStream oos = new ObjectOutputStream(
                            new FileOutputStream("egitilmis.model"));
 oos.writeObject(cls);
 oos.flush();
 oos.close();

Yukarıdaki kodda, öncelikle bir sınıflandırıcı nesnesini tanımlıyoruz. Ardından dosyayı okuyup bir instance nesnesine transfer edip cls.buildClassifier metodu ile verilen inst değişkenindeki nesneyi eğitim için kullanıyoruz. Artık bu satırdan sonra elimizde eğitilmiş bir model bulunuyor. Ardından eğitilmiş ve eğitim verilerini tutan modelimizi serileyerek bir dosyaya kaydediyoruz. Bunun için JAVA’nın sınıflarından birisi olan ObjectOutputStream sınıfının ve bu sınıfın writeObject metodunu kullanıyoruz. Son olarak açtığımız akışları (stream) kapatıyoruz.

Alternatif olarak WEKA’nın 3.3.5 versiyonundan sonraki sürümleri için weka’nın içindeki bir sınıfı da kullanabiliriz:

weka.core.SerializationHelper.write("egitilmis.model", cls);

Yukarıdaki tek satır ile, ilk kodda bulunan ObjectOutputStream sınıfının tanımı ve writeObject metodunun çağırılması işlemlerini tek bir adımda yapılabilir.

Şimdi gelelim oluşturduğumuz bu modelin test için kullanılmasına. Bunun için tersserileme (deserialization) uygulayarak daha önce dizgiye (string) çevirdiğimiz nesneyi, tekrar dizgiden (string) nesneye geri çevireceğiz. JAVA’nın sınıflarını kullanarak aşağıdaki şekilde bu kodu yazabiliriz:

ObjectInputStream ois = new ObjectInputStream(
                           new FileInputStream("egitim.model"));
 Classifier cls = (Classifier) ois.readObject();
 ois.close();

Yukarıda, ObjectInputStream sınıfını ve bu sınıfın readObject metodunu kullanarak, daha önce kaydettiğimiz egitim.model dosyasından veriyi okuyarak cls isimli nesneye yükledik. Artık bu nesneyi test kümesini sınıflandırmak için kullanabiliriz. Veya yine alternatif olarak, weka’nın 3.3.5 versiyonu sonrası sürümleri ile gelen aşağıdaki satırı kullanabiliriz:
"""

doc3 = u"""
Bu yazının amacı, bilgisayar bilimleri ve iş zekası (business intelligence) gibi disiplinlerin ortak çalışma alanlarından olan veri madenciliği (data mining) konusunda kullanılan metotlardan birisi olan sınıflandırma (classification) kavramını açıklamaktır.

Sınıflandırma kavramı, basitçe bir veri kümesi (data set) üzerinde tanımlı olan çeşitli sınıflar arasında veriyi dağıtmaktır. Sınıflandırma algoritmaları, verilen eğitim kümesinden bu dağılım şeklini öğrenirler ve daha sonra sınıfının belirli olmadığı test verileri geldiğinde doğru şekilde sınıflandırmaya çalışırlar.

Veri kümesi üzerinde verilen bu sınıfları belirten değerlere etiket (label) ismi verilir ve gerek eğitim gerekse test sırasında verinin sınıfının belirlenmesi için kullanılırlar.

Konunun daha kolay anlaşılabilmesi için bir örnek üzerinden anlatmaya çalışalım.

Örneğin aşağıdaki şekilde öğrencilerden toplanan bir veri kümemiz bulunsun.


Örneğin problemimizi şu şekilde tanımlayalım. Bir sınıflandırıcı yöntemimiz, yukarıdaki kümeye bakarak verilen yaş boy ve kilo değerlerine göre bir öğrencinin cinsiyetini öğrenecek olsun. Yani yukarıdaki veri kümesini bir eğitim kümesi olarak kullanacağız. Ardından gelen yeni bir kayıt için, yaş, boy ve kilo değerleri verildiğinde, sınıflandırıcımız cinsiyetini otomatik olarak tahmin edecek (prediction).

Çok sayıdaki sınıflandırma algoritmalarından basit birini seçelim. Diyelim ki sınıflandırma algoritmamız, verilen etiketteki değerlerin ortalamasını alacak ve bu ortalama değer, öğrendiği değer olacak. Ardından gelen test değerleri için bulmuş olduğu ortalamaya uzaklığına bakacak ve kime yakınsa o etiketten kabul edecek.

Yukarıdaki veri kümesini iki sınıf için ikiye bölelim ve ortalama değerlerini alalım:

Erkekler için öğrenme işlemimiz aşağıda verilmiştir:

Sonuç olarak algoritmamız erkekler için (20,179.5, 78.5) değerlerini öğrenirken, kızlar için (20.66, 167.33, 54) değerlerini öğreniyor. Diyelim ki yeni gelen ve test edilmesini istediğimiz değer de, yaş : 21, boy: 165, kilo 60 değerlerinde bir kişi olsun. Şimdi algoritmamız öğrendiği değerlere göre bu yeni gelen kişinin cinsiyetini tahmin etmeye çalışacak. Basitçe her değere olan mesafeyi hesaplayacak (burada da çok farklı mesafe hesaplama algoritmaları olmasına karşılık biz yine amacımız temel kavramlar olduğu için konuyu basit tutarak öklit mesafesi (euclidean distance) kullanalım).

Algoritmamızın Erkek tanımından öğrendiği değerler ile yeni gelen kişinin mesafesini hesaplayalım:

Benzer şekilde kızlar için öğrendiğimiz değere olan mesafesini hesaplamaya çalışalım.

Buna göre algoritmamızın verdiği değer, erkeklere olan mesafesinin 23.72 olduğu ve kızlara olan mesafesinin 6.44 olduğudur. Demek ki algoritmamız yeni gelen kişiyi kız sınıfından tanımlamıştır.

İşte sınıflandırma algoritmalarının çalışması, yukarıda da anlatıldığı gibi en basit anlamıyla, 2 aşamadan oluşur.

    Eğitim verisi üzerinden öğrenme

    Öğrenilen değerlerle test verisi üzerinde sınıflandırma

Ancak veri madenciliği ve iş zekası çalışmalarında sınıflandırma sadece çalışmanın bir türü ve ufak bir parçası olmaktadır.
"""



example = tfidf('tr',[doc1,doc2,doc3],['a','b','c'])
example.get_tfidf()