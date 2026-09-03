#include <bits/stdc++.h>
using namespace std;

#define int long long

void update(vector<int> &tree, int index, int start, int end, int to, int val){
    if (start > end) return;

    if ((start < to && end < to) || (start > to && end > to))
        return;

    if (start == to && end == to){
        tree[index] += val;   // CHANGE 1
        return;
    }

    int mid = (start + end) / 2;

    update(tree, 2 * index, start, mid, to, val);
    update(tree, 2 * index + 1, mid + 1, end, to, val);

    tree[index] = tree[2 * index] + tree[2 * index + 1];
}


int range(vector<int> &tree, int index, int start, int end, int from, int to){
    if (start > end) return 0;

    if (start > to || end < from)
        return 0;

    if (from <= start && end <= to)
        return tree[index];

    int mid = (start + end) / 2;

    // CHANGE 2: correct child ranges
    int one = range(tree, 2 * index, start, mid, from, to);
    int two = range(tree, 2 * index + 1, mid + 1, end, from, to);

    return one + two;
}


signed main() {

    int n, k;
    cin >> n >> k;

    vector<int> ans(n);

    for (int i = 0; i < n; i++)
        cin >> ans[i];

    // int maxi = *max_element(ans.begin(), ans.end());

    vector<int> tree(4 * n + 2);

    vector<int> finale;
    
    vector <int> useful = ans;
    sort(useful.begin(), useful.end());
    
    useful.erase(unique(useful.begin(), useful.end()), useful.end());
    int maxi = useful.size();
    for (int i = 0; i<n; i++){
        ans[i] = std::lower_bound(useful.begin(), useful.end(), ans[i])+1-useful.begin();
    }
    
    
    
    int one = 0;
    
    for (int i = 0; i < k; i++) {
    update(tree, 1, 1, maxi, ans[i], 1);
        
        one += range(tree, 1, 1, maxi, ans[i] + 1, maxi);

        // Add ans[i]
    
    }

    finale.push_back(one);

    int left = 0;
    int right = k;

    while (right < n) {

        // Remove ans[left]
        // Count elements smaller than it
        int two = range(tree, 1, 1, maxi, 1, ans[left] - 1);

        one -= two;

        update(tree, 1, 1, maxi, ans[left], -1);

        // Add ans[right]
        // Count existing elements greater than it
        int three = range(tree, 1, 1, maxi, ans[right] + 1, maxi);

        one += three;

        update(tree, 1, 1, maxi, ans[right], 1);

        finale.push_back(one);

        left++;
        right++;
    }

    for (int i = 0; i < finale.size(); i++)
        cout << finale[i] << " ";

    cout << endl;
}