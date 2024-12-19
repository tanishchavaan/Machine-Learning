#include
using namespace std;

string solution(int n, int m) {
    if (n < m) {
        return "lesser";
    } else if (n == m) {
        return "equal";
    } else {
        return "greater";
    }
}
hi , i am
int main() {
    int n, m;
    cin >> n >> m;
    cout << solution(n, m) << endl;
    return 0;
}
