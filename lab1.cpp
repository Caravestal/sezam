#include <iostream>
#include <queue>
#include <chrono>
#include <thread>
#include <mutex>
#include <vector>

using namespace std;

int counter = 10;

void z1_zwieksz_licznik()
{
    for (int i = 0; i < 10000; i++)
    {
        counter += 1;
    }
    cout << counter <<endl;
}

void z1_zmniejsz_licznik()
{
    for (int i = 0; i < 10000; i++)
    {
        counter -= 1;
        cout << counter<<endl;
    }
}

void z1()
{
    thread watek1(z1_zwieksz_licznik);
    thread watek2(z1_zmniejsz_licznik);
    watek1.join();
    watek2.join();
    cout << counter << endl;
}

queue<int> kolejka;
bool zakonczono = false;
mutex blokada;

void z2_producent()
{
    for (int i = 0;i < 50;i++)
    {
        while (kolejka.size() >= 10)
        {
            this_thread::sleep_for(chrono::seconds(1));
        }
        blokada.lock();
        kolejka.push(rand()%100);
        blokada.unlock();
        cout << "Wstawiono do kolejki: " << kolejka.front() << endl;
    }
    zakonczono = true;
}

void z2_konsument()
{
    while (zakonczono != true || kolejka.size() > 0)
    {
        while (kolejka.size() == 0)
        {
            this_thread::sleep_for(chrono::seconds(1));
        }
        blokada.lock();
        int liczba_z_kolejki = kolejka.front();
        kolejka.pop();
        blokada.unlock();
        int potega = rand() % 5;
        cout << "liczba z kolejki: " << liczba_z_kolejki << " potega: " << potega << " = " << pow(liczba_z_kolejki, potega) << endl;
    }
}

void z2()
{
    thread watek1(z2_producent);
    this_thread::sleep_for(chrono::seconds(1));
    thread watek2(z2_konsument);
    watek1.join();
    watek2.join();
}

int n = 100;

void z3_dodaj(int* a,int* b,int* c,int i)
{
    c[i] = a[i] + b[i];
    cout << i << endl;
}

void z3()
{
    int* a = new int[n];
    int* b = new int[n];
    int* c = new int[n];
    vector<thread> vector_watkow;
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
        vector_watkow.emplace_back(z3_dodaj,a,b,c,i);
    }
    for (int i = 0; i < n; i++)
    {
        vector_watkow[i].join();
    }
}

void z4_watki(bool czy_dodac, int* a, int* b, int* c)
{
    if (czy_dodac)
    {
        for (int i = 0;i < n;i++) 
        {
            c[i] = a[i] + b[i];
        }
        return;
    }
    for (int i = 0;i < n;i++)
    {
        c[i] = a[i] * b[i];
    }
}

void z4()
{
    int* a = new int[n];
    int* b = new int[n];
    int* c = new int[n];
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }
    thread watek1(z4_watki, true, a, b, c);
    thread watek2(z4_watki, false, a, b, c);
    watek1.join();
    watek2.join();
}

int main()
{
    //z1();
    //z2();
    //z3();
    //z4();
    return 0;
}

