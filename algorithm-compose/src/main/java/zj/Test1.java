package zj;

public class Test1 {

    public static void main(String[] args) {
        int[] nums = new int[]{10, 9, 2, 5, 3, 7, 101, 18};
        int res = seq(nums);
        System.out.println(res);
    }

    public static int seq(int[] nums) {
        int length = nums.length;
        if (length == 0) {
            return 0;
        }
        int[] dp = new int[length + 1];
        for (int i = 1; i < nums.length; i++) {
            for (int j = 1; j <= i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[j - 1] + 1, dp[j - 1]);
                }
            }
        }
        return dp[length - 1];
    }
}
