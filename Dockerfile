FROM python:3.10-slim

WORKDIR /app

# Копируем файлы проекта
COPY . .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Настраиваем параметры Streamlit
RUN mkdir -p /root/.streamlit
RUN echo "\
[server]\n\
port = 7050\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > /root/.streamlit/config.toml

# Определяем ограничения ресурсов (в Docker Compose или при запуске контейнера)
# --memory=8g --cpus=6 --storage-opt size=50G

# Определяем порт, который будет использоваться
EXPOSE 7050

# Запускаем Streamlit
CMD ["streamlit", "run", "app2.py", "--server.port=7050", "--server.address=0.0.0.0"]

# Примечание: Для установки ограничений ресурсов используйте следующую команду при запуске:
# docker run -p 7050:7050 --memory=8g --cpus=6 --storage-opt size=50G имя_образа
