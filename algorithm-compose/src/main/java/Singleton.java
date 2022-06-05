//public class Singleton {
//    //懒汉式-线程不安全
//    private  static Singleton uniqueInsstance;
//
//    private Singleton(){
//    }
//
//    public static Singleton getUniqueInsstance(){
//        if(uniqueInsstance == null){
//            uniqueInsstance = new Singleton();
//        }
//        return uniqueInsstance;
//    }
//
//    //懒汉式-线程安全
//    public static synchronized Singleton getUniqueInstance(){
//        if(uniqueInsstance == null){
//            uniqueInsstance = new Singleton();
//        }
//        return uniqueInsstance;
//    }
//
//    //饿汉式-线程安全
//    private static Singleton uniqueInstance = new Singleton();
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