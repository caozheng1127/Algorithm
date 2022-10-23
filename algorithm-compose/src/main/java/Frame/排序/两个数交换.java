package Frame.排序;

/**
 * @Description
 * @date 2018/8/15 9:09
 */
public class 两个数交换 {
    public static void main(String[] args) {
        int a=10;
        int b=20;
        a=a^b;
        b=a^b;
        a=a^b;
        System.out.println("a="+a+"b="+b);
    }
}
