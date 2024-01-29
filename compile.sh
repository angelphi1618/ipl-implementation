#Compilar una vez
icpx main.cpp -fsycl


#Compilar dos veces
#icpx -fPIC -c main.cpp -fsycl
#icpx -fPIC main.o -fsycl