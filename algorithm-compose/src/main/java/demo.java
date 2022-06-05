import java.lang.reflect.Array;
import java.util.*;

public class demo {
    public static void main(String[] args) {
    }

    //二分查找
    public int search(int key, int[] array) {
        int l = 0, h = array.length - 1;
        while (l <= h) {
            int mid = l + (h - l) / 2;
            if (key == array[mid]) return mid;
            if (key < array[mid]) h = mid - 1;
            else l = mid + 1;
        }
        return -1;
    }

    //贪心
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int i = 0, j = 0;
        while (i < g.length && j < s.length) {
            if (g[i] <= s[j]) i++;
            j++;
        }
        return i;
    }

    //双指针
    public int[] twoSum(int[] numbers, int target) {
        int i = 0, j = numbers.length - 1;
        while (i < j) {
            int sum = numbers[i] + numbers[j];
            if (sum == target) return new int[]{i + 1, j + 1};
            else if (sum < target) i++;
            else j--;
        }
        return null;
    }

    //反转字符串中的元音字符
    private HashSet<Character> vowels = new HashSet<>(Arrays.asList('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'));

    public String reverseVowels(String s) {
        if (s.length() == 0) return s;
        int i = 0, j = s.length() - 1;
        char[] result = new char[s.length()];
        while (i <= j) {
            char ci = s.charAt(i);
            char cj = s.charAt(j);
            if (!vowels.contains(ci)) {
                result[i] = ci;
                i++;
            } else if (!vowels.contains(j)) {
                result[j] = cj;
                j--;
            } else {
                result[i] = ci;
                result[j] = cj;
                i++;
                j--;
            }
        }
        return new String(result);
    }

    //两数平方和
    public boolean judgeSquareSum(int c) {
        int left = 0, right = (int) Math.sqrt(c);
        while (left <= right) {
            int powSum = left * left + right * right;
            if (powSum == c) return true;
            else if (powSum > c) right--;
            else left++;
        }
        return false;
    }

    //回文字符串
    public boolean validPalindrome(String s) {
        int i = 0, j = s.length() - 1;
        while (i < j) {
            if (s.charAt(i) != s.charAt(j)) {
                return isPalindrome(s, i, j - 1) || isPalindrome(s, i + 1, j);
            }
            i++;
            j--;
        }
        return true;
    }

    private boolean isPalindrome(String s, int l, int r) {
        while (l < r) {
            if (s.charAt(l) != s.charAt(r))
                return false;
            l++;
            r--;
        }
        return true;
    }

    //归并两个有序数组
    public void merge(int[] num1, int m, int[] num2, int n) {
        int i = m - 1, j = n - 1;
        int idx = m + n - 1;
        while (i >= 0 || j >= 0) {
            if (i < 0) num1[idx] = num2[j--];
            else if (j < 0) num1[idx] = num2[j--];
            else if (num1[i] > num2[j]) num1[idx] = num1[i--];
            else num1[idx] = num2[j--];
            idx--;
        }
    }

    //最长子序列
    public String findLongestWord(String s, List<String> d){
        String ret = "";
        for(String str : d){
            for (int i = 0, j=0;i<s.length()&&j<str.length();i++){
                if(s.charAt(i) == str.charAt(j)) j++;
                if(j == str.length()){
                    if(ret.length()<str.length()||(ret.length()==str.length()&&ret.compareTo(str)>0)){
                        ret =str;
                    }
                }
            }
        }
        return ret;
    }
}