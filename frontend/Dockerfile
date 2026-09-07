FROM node:20-alpine

WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED=1 \
    PORT=3001

ARG NEXT_PUBLIC_BACKEND_WS=ws://localhost:8000/ws/stream
ENV NEXT_PUBLIC_BACKEND_WS=$NEXT_PUBLIC_BACKEND_WS

COPY package.json package-lock.json* ./

RUN npm install

COPY . .

RUN npm run build

ENV NODE_ENV=production

EXPOSE 3001

CMD ["npx", "next", "start", "-p", "3001"]
