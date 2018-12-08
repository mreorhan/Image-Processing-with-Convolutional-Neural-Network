YZM 4213- Final Project Report

# Image Processing(Fruits) with Convolutional Neural Network


1. I.Özet

     Yiyecekler insanların yaşamının çok önemli bir parçasıdır. Fakat bazı nedenlerle (görme kaybı) her zaman yiyeceğe ulaşmayı zorlaştırır. Burada amaç ilk olarak daha küçük bir küme olan meyveleri tanımlamaktır.

Convolutional Neural Network ile resim işleme ve resim tahminlemesi yapılabilmektedir. Bunu yaparken resme çeşitli fitreler uygulayıp, katmanlar eklenir. Filtrelerde kaydırma işlemi yapılarak (stride) resimlerdeki karmaşıklığın (detayların) azaltılması sağlanır. Ayrıca parametre sayısını azaltmak için de pooling yöntemi kullanılır. Filtrelerin uygulanması sonucunda aktivasyon haritaları ortaya çıkar. Buradaki uygulamada amaç derin öğrenme metotları kullanılarak ağdaki kaybı minimize etmektir. Bunun için çeşitli aktivasyon fonksiyonları kullanarak, epoch sayısını değiştirerek öğrenme ve test süreci geçirilecektir. Aşırı öğrenme (overfitting) olmasını önlemek için Dropout yöntemi kullanılacaktır. Elde edilen sonuçlar grafik olarak belirtilecektir.

1. II.Anahtar Sözcükler

**docx; Times New Roman 10pt; Deep Learning,**** Image Classification, Convolutional Neural Network, Keras, Image Processing**

1. III.Giriş

Konvolüsyonel nöral ağ (CNN veya ConvNet), yaygın olarak görsel imgelerin analiz edilmesinde kullanılan derin, ileri beslemeli yapay sinir ağlarıdır.

CNN&#39;ler, minimum ön işlem gerektirecek şekilde tasarlanmış çok katmanlı algılayıcı çeşitlerini kullanırlar.

CNN&#39;ler, diğer görüntü sınıflandırma algoritmalarına kıyasla nispeten az ön işlem kullanırlar. Bu, ağın geleneksel algoritmalarda el yapımı olan filtreleri öğreneceği anlamına gelir



1. Şekil. 1.CNN mimarileri

     CNN&#39;de birçok mimari bulunup farklı katman sayıları, farklı aktivasyon fonksiyonları bulunmaktadır.

_&quot;CNN&#39;de AlexNet mimarisi bu alanda araştırma patlamasına neden olan bir mimari olup relu aktivasyon fonksiyonunu da ilk kullanan mimaridir.&quot;_

CNN&#39;in görüntü ve video tanıma, öneri sistemleri, görüntü sınıflandırması, medikal görüntü analizi ve doğal dil işleme gibi uygulamaları vardır.

1. Şekil. 2. Örnek bir CNN&#39;de kullanılan katman yapısı

1. IV.Önceki Çalışmalar

     KURTULDU (2002), &quot;Hücresel sinir ağları (CNN) modelleri&quot; Yapay nöral ağları, son 10 yıldır hızla artan bir ilginin odağı olmaktadır. Günümüzdeki kullanılan bilgisayarlar dijital devreler ile gerçeklenmesi çok zor olan bazı tür fonksiyonları insanlar ve hayvanlar kolaylıkla gerçekleştirilebilmektedirler. Bilim adamları bir yandan insan beyninin çalışmasını çözmeye çalışırken diğer yandan da edindikleri bilgilere yeni ve daha güçlü bilgisayarlar, makineler yapmaya çalışmaktadırlar. Bu tezde yapay nöral ağlar hakkında bazı genel bilgiler verildikten sonra nöral ağların özel bir türü olan Hücresel Nöral Ağlar konusu incelenecektir. Hücresel nöral ağlar, nöral ağ teorisindeki bazı boşluklar doldurulacak şekilde geliştirilmiş ve görüntü işleme ve tanıma problemlerine başarıyla uygulanmış analog, paralel devrelerdir. HNA&#39;ların teorik yapısı incelendikten sonra çeşitli alanlardaki uygulamalarına da yer verilecektir. Bu tezde hücresel ağlar ile orijinal bir uygulama da gerçekleştirilecek, işaret işlemenin önemli problemlerinden biri olan İşaret Ayrıştırma problemi de yeni bir küme kalıbı bulunarak çözülmüştür.

     KİTAKYUSHU, Japan; Nishi, Toshiki; Kurogi, Shuichi; Matsuo, Kazuya, Grading Fruits and Vegetables Using RGB-D Images and Convolutional Neural Network, &quot;This thesis presents a method for grading fruits and vegetables by means of using RGB-D (RGB and depth) images and convolutional neural network (CNN). Here, we focus on grading according to the size of objects. First, the method transforms positions of pixels in RGB image so that the center of the object in 3D space is placed at the position equidistant from the focal point by means of using the corresponding depth&quot;

     XU, (2016), &quot;Deep Convolutional Networks for Image Classification&quot; Deep neural networks, particularly deep convolutional networks, have recently contributed great improvements to end-to-end learning quality for this problem. In this thesis I address two questions: first, how best to design the architecture of a convolutional neural network for image classification; and second, how to improve the activation functions used in convolutional neural networks. I review the history of convolutional network architectures, then propose an efficient network structure named &quot;TinyNet&quot; that reduces network size while preserving state of the art image classification performance.

     SOHONY &quot;Fruits-360- Transfer Learning using Keras&quot;, Bu uygulama sadece test amaçlı yapılmış olup akademik bilgi içermemektedir.

     Fruits 360 veri seti ile ilgili akademik çalışma bulunmamakla beraber benzer çalışmalar yukarıda belirtilmiştir.

1. V.Yöntem

     Veri seti olarak Fruits 360 kullanılmaktadır. Bu veri setinde eğitim için 41322 meyve resmi, test için ise 13877 meyve resmi bulunmaktadır. Toplamda 81 kategoride sınıflandırılmış meyveye ait kayıt bulunmaktadır. Veri setinde önceden işlenmiş 100X100 boyutunda jpg tipinde resimler kullanılmıştır. Ayrıca sınıflandırılması klasör olarak yapılıp bu şekilde kullanılmaktadır. Proje Pycharm ide&#39;si üzerinde Python 3.6.5 sürümü ile Keras üzerinden gerçekleştirilmektedir. Katmanlar üzerinde aktivasyon fonksiyonlarından Adam, Adadelta, Adamax, Stokastik Gradyan İniş ve Relu kullanılmıştır. Farklı işlemlere tabi tutulduğundan Dropout (0.0.1-0.05 arası değerler) yöntemi de denenmiştir. Padding sabit tutulmuş olup stride(3,3) tür. Resimler başlangıçta 64X64 olacak şekilde yeniden boyutlandırılıp cv2 kütüphanesi kullanılarak BGR2RGB filtresi uygulanmıştır. Pooling yöntemi olarak max pooling(2,2) kullanılmıştır. Grafikler için python kütüphanesi olan matpotlib kullanılmıştır.

1. VI.Deneysel Sonuçlar

1. TABLO I. Katmanların ağ üzerindeki etkisi

| **Epoch Sayısı** | **Kullanılan Araçlar ve Sonuçlar** |
| --- | --- |
| **Katman Sayısı** | **Optimizasyon Aracı** | **Loss** | **Accuracy** |
| **2** | 4 | Adadelta | 0.1159 | 0.9681 |
| **2** | 5 | Adadelta | 0.4002 | 0.8726 |

1. Keras sonuçları toplam değil son epoch baz alınmıştır. Katmanlarda (çıkış hariç) relu fonksiyonu kullanılmıştır.

     Yukarıdaki sorun CNN&#39;de &quot;Resudial Block&quot; ile aşılmaya çalışılmıştır.

1. Şekil. 3.&quot;Resudial Block&quot; Yapısı

1. Şekil. 4.Dropout = 0.01 Kullanımı

##### Kaynaklar

1. [1]Fruits 360 dataset: A dataset of images containing fruits [https://github.com/Horea94/Fruit-Images-Dataset](https://github.com/Horea94/Fruit-Images-Dataset)
2. [2]Alex Krizhevsky Ilya Sutskever Geoffrey E. Hinton, &quot;ImageNet Classification with Deep Convolutional Neural Networks&quot; , (NIPS 2012)
3. [3]Python 3.6.5, https://www.python.org
4. [4]Pycharm community edition 2018, https://www.jetbrains.com/pycharm
5. [5]Open Access Theses and Dissertations https://oatd.org/
6. [6]Ulusal Tez Merkezi https://tez.yok.gov.tr/UlusalTezMerkezi/
7. [7]Matplotlib https://matplotlib.org/tutorials/index.html
8. [8]Grafikler https://www.udemy.com/derin-ogrenmeye-giris/
9. [9]Deep Learning article on Wikipedia. https://en.wikipedia.org/ wiki/Deep\_learning.
