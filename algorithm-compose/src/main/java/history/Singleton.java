package history;//public class history.Singleton {
//    //懒汉式-线程不安全
//    private  static history.Singleton uniqueInsstance;
//
//    private history.Singleton(){
//    }
//
//    public static history.Singleton getUniqueInsstance(){
//        if(uniqueInsstance == null){
//            uniqueInsstance = new history.Singleton();
//        }
//        return uniqueInsstance;
//    }
//
//    //懒汉式-线程安全
//    public static synchronized history.Singleton getUniqueInstance(){
//        if(uniqueInsstance == null){
//            uniqueInsstance = new history.Singleton();
//        }
//        return uniqueInsstance;
//    }
//
//    //饿汉式-线程安全
//    private static history.Singleton uniqueInstance = new history.Singleton();
//}

//双重校验锁-线程安全
public class Singleton {
    private volatile static Singleton uniqueInstance;

    private Singleton() {
    }

    public static Singleton getUniqueInstance() {
        if (uniqueInstance == null) {
            synchronized (Singleton.class) {
                if (uniqueInstance == null) {
                    uniqueInstance = new Singleton();
                }
            }
        }
        return uniqueInstance;
    }
}