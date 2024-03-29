package Frame.排序;

/**
 * @Description
 * @date 2018/8/19 13:49
 */
public class ShellSort {
    public void shellSort(int[] a) {
        int temp = 0;
        double d1 = a.length;
        while (true) {

            d1 = Math.ceil(d1 / 2);
            int d = (int) d1;

            for (int x = 0; x < d; x++) {
                for (int i = x + d; i < a.length; i += d) {
                    int j = i - d;
                    temp = a[i];
                    for (; j >= 0 && temp < a[j]; j -= d) {
                        a[j + d] = a[j];
                    }
                    a[j + d] = temp;
                }
            }
            if (d == 1) {
                break;
            }
        }
        for (int i = 0; i < a.length; i++)
            System.out.print(a[i] + "\t");
    }
}
