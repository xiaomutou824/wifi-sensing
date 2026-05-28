/*
 * ESP32-S3 B: STA CSI receiver.
 *
 * This board connects to CSI_AP, enables CSI, and pings the AP gateway.
 * AP replies are stable AP-to-STA packets used for CSI collection.
 *
 * Output format:
 * CSI_DATA,node_id,seq,local_time_us,rx_timestamp_us,src_mac,dst_mac,
 * first_word_invalid,rx_seq,payload_len,rssi,channel,secondary_channel,rate,
 * sig_mode,mcs,cwb,stbc,sgi,noise_floor,ant,sig_len,rx_state,csi_len,
 * csi_bytes...
 */

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "esp_check.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/ip_addr.h"
#include "nvs_flash.h"
#include "ping/ping_sock.h"

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT BIT1
#define WIFI_MAXIMUM_RETRY 10
#define MAC_STR "%02x:%02x:%02x:%02x:%02x:%02x"
#define MAC_ARG(mac) (mac)[0], (mac)[1], (mac)[2], (mac)[3], (mac)[4], (mac)[5]
#define CSI_LINE_BUFFER_SIZE 2048

static const char *TAG = "csi_sta";
static EventGroupHandle_t wifi_event_group;
static int retry_count;
static uint32_t csi_sequence;
static uint8_t connected_ap_bssid[6];
static bool connected_ap_bssid_valid;
static uint8_t sta_mac[6];
static bool sta_mac_valid;
static esp_netif_ip_info_t wifi_ip_info;
static bool wifi_ip_info_valid;
static esp_ping_handle_t gateway_ping;
static volatile uint32_t csi_callback_count;
static volatile uint32_t csi_printed_count;
static volatile uint32_t csi_filtered_count;
static volatile uint32_t csi_filtered_ap_count;
static volatile uint32_t csi_filtered_dst_count;
static volatile uint32_t csi_filtered_sig_mode_count;
static volatile uint32_t csi_null_count;
static volatile uint32_t csi_rx_seq_gap_count;
static volatile uint32_t ping_success_count;
static volatile uint32_t ping_timeout_count;
static uint16_t last_rx_seq;
static bool last_rx_seq_valid;

static int append_text(char *buf, size_t buf_size, int offset,
                       const char *fmt, ...) {
  if (offset < 0 || (size_t)offset >= buf_size) {
    return -1;
  }

  va_list args;
  va_start(args, fmt);
  const int written = vsnprintf(buf + offset, buf_size - (size_t)offset, fmt, args);
  va_end(args);
  if (written < 0 || (size_t)written >= buf_size - (size_t)offset) {
    return -1;
  }
  return offset + written;
}

static void init_nvs(void) {
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);
}

static const char *csi_mode_name(void) {
#if CONFIG_CSI_MODE_RAW_HTLTF
  return "raw_htltf";
#elif CONFIG_CSI_MODE_ROUTER_COMPATIBLE_LLTF
  return "router_compatible_lltf";
#elif CONFIG_CSI_MODE_MERGED_STABLE
  return "merged_stable";
#elif CONFIG_CSI_MODE_RESEARCH_FULL
  return "research_full";
#else
  return "unknown";
#endif
}

static bool should_print_csi_frame(const wifi_csi_info_t *info) {
  const wifi_pkt_rx_ctrl_t *rx = &info->rx_ctrl;

#if CONFIG_CSI_FILTER_AP_BSSID
  if (!connected_ap_bssid_valid ||
      memcmp(info->mac, connected_ap_bssid, sizeof(connected_ap_bssid)) != 0) {
    csi_filtered_ap_count++;
    return false;
  }
#endif

#if CONFIG_CSI_FILTER_STA_DST_MAC
  if (!sta_mac_valid || memcmp(info->dmac, sta_mac, sizeof(sta_mac)) != 0) {
    csi_filtered_dst_count++;
    return false;
  }
#endif

#if CONFIG_CSI_FILTER_HT_FRAMES
  if (rx->sig_mode != 1) {
    csi_filtered_sig_mode_count++;
    return false;
  }
#endif

  return true;
}

static void sta_event_handler(void *arg, esp_event_base_t event_base,
                              int32_t event_id, void *event_data) {
  (void)arg;
  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
    esp_wifi_connect();
  } else if (event_base == WIFI_EVENT &&
             event_id == WIFI_EVENT_STA_DISCONNECTED) {
    wifi_ip_info_valid = false;
    connected_ap_bssid_valid = false;
    if (retry_count < WIFI_MAXIMUM_RETRY) {
      retry_count++;
      ESP_LOGW(TAG, "Wi-Fi disconnected, retrying (%d/%d)", retry_count,
               WIFI_MAXIMUM_RETRY);
      esp_wifi_connect();
    } else {
      xEventGroupSetBits(wifi_event_group, WIFI_FAIL_BIT);
    }
  } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
    retry_count = 0;
    wifi_ip_info = event->ip_info;
    wifi_ip_info_valid = true;
    ESP_LOGI(TAG, "STA connected, IP: " IPSTR ", gateway: " IPSTR,
             IP2STR(&event->ip_info.ip), IP2STR(&event->ip_info.gw));
    xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
  }
}

static esp_err_t wifi_init_sta(void) {
  wifi_event_group = xEventGroupCreate();
  ESP_RETURN_ON_FALSE(wifi_event_group != NULL, ESP_ERR_NO_MEM, TAG,
                      "Failed to create Wi-Fi event group");

  ESP_RETURN_ON_ERROR(esp_netif_init(), TAG, "esp_netif_init failed");
  ESP_RETURN_ON_ERROR(esp_event_loop_create_default(), TAG,
                      "event loop init failed");
  esp_netif_create_default_wifi_sta();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_RETURN_ON_ERROR(esp_wifi_init(&cfg), TAG, "esp_wifi_init failed");
  ESP_RETURN_ON_ERROR(esp_event_handler_instance_register(
                          WIFI_EVENT, ESP_EVENT_ANY_ID, &sta_event_handler,
                          NULL, NULL),
                      TAG, "register Wi-Fi event handler failed");
  ESP_RETURN_ON_ERROR(esp_event_handler_instance_register(
                          IP_EVENT, IP_EVENT_STA_GOT_IP, &sta_event_handler,
                          NULL, NULL),
                      TAG, "register IP event handler failed");

  wifi_config_t wifi_config = {0};
  strlcpy((char *)wifi_config.sta.ssid, CONFIG_CSI_AP_SSID,
          sizeof(wifi_config.sta.ssid));
  strlcpy((char *)wifi_config.sta.password, CONFIG_CSI_AP_PASSWORD,
          sizeof(wifi_config.sta.password));
  wifi_config.sta.channel = CONFIG_CSI_AP_CHANNEL;
  wifi_config.sta.threshold.authmode = strlen(CONFIG_CSI_AP_PASSWORD) == 0
                                           ? WIFI_AUTH_OPEN
                                           : WIFI_AUTH_WPA2_PSK;
  wifi_config.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;

  ESP_RETURN_ON_ERROR(esp_wifi_set_mode(WIFI_MODE_STA), TAG,
                      "set Wi-Fi STA mode failed");
  ESP_RETURN_ON_ERROR(esp_wifi_set_config(WIFI_IF_STA, &wifi_config), TAG,
                      "set STA config failed");
  ESP_RETURN_ON_ERROR(esp_wifi_set_ps(WIFI_PS_NONE), TAG,
                      "disable STA power save failed");
  ESP_RETURN_ON_ERROR(esp_wifi_start(), TAG, "start STA failed");

  ESP_LOGI(TAG, "Connecting to SoftAP: ssid=%s, channel=%d",
           CONFIG_CSI_AP_SSID, CONFIG_CSI_AP_CHANNEL);
  EventBits_t bits = xEventGroupWaitBits(
      wifi_event_group, WIFI_CONNECTED_BIT | WIFI_FAIL_BIT, pdFALSE, pdFALSE,
      portMAX_DELAY);
  ESP_RETURN_ON_FALSE(bits & WIFI_CONNECTED_BIT, ESP_FAIL, TAG,
                      "failed to connect to SoftAP");

  wifi_ap_record_t ap_info = {0};
  ESP_RETURN_ON_ERROR(esp_wifi_sta_get_ap_info(&ap_info), TAG,
                      "get AP info failed");
  memcpy(connected_ap_bssid, ap_info.bssid, sizeof(connected_ap_bssid));
  connected_ap_bssid_valid = true;
  ESP_RETURN_ON_ERROR(esp_wifi_get_mac(WIFI_IF_STA, sta_mac), TAG,
                      "get STA MAC failed");
  sta_mac_valid = true;
  ESP_LOGI(TAG, "Connected AP BSSID: " MAC_STR ", channel=%u, rssi=%d",
           MAC_ARG(connected_ap_bssid), ap_info.primary, ap_info.rssi);
  ESP_LOGI(TAG, "STA MAC: " MAC_STR, MAC_ARG(sta_mac));
  return ESP_OK;
}

static void csi_rx_callback(void *ctx, wifi_csi_info_t *info) {
  (void)ctx;
  csi_callback_count++;
  if (info == NULL || info->buf == NULL) {
    csi_null_count++;
    return;
  }
  if (!should_print_csi_frame(info)) {
    csi_filtered_count++;
    return;
  }

  const wifi_pkt_rx_ctrl_t *rx = &info->rx_ctrl;
  const int64_t now_us = esp_timer_get_time();
  const uint32_t seq = csi_sequence++;
  if (last_rx_seq_valid && info->rx_seq != ((last_rx_seq + 1) & 0x0FFF)) {
    csi_rx_seq_gap_count++;
  }
  last_rx_seq = info->rx_seq;
  last_rx_seq_valid = true;
  csi_printed_count++;

  char line[CSI_LINE_BUFFER_SIZE];
  int offset = append_text(
      line, sizeof(line), 0,
      "CSI_DATA,%d,%" PRIu32 ",%" PRId64 ",%" PRIu32 "," MAC_STR ","
      MAC_STR ",%u,%u,%u,%d,%u,%u,%u,%u,%u,%u,%u,%u,%d,%u,%u,%u,%u",
      CONFIG_CSI_NODE_ID, seq, now_us, (uint32_t)rx->timestamp,
      MAC_ARG(info->mac), MAC_ARG(info->dmac),
      info->first_word_invalid ? 1 : 0, info->rx_seq, info->payload_len,
      rx->rssi, rx->channel, rx->secondary_channel, rx->rate, rx->sig_mode,
      rx->mcs, rx->cwb, rx->stbc, rx->sgi, rx->noise_floor, rx->ant,
      rx->sig_len, rx->rx_state, info->len);

#if CONFIG_CSI_PRINT_RAW_IQ
  for (int i = 0; i < info->len && offset >= 0; i++) {
    offset = append_text(line, sizeof(line), offset, ",%d", info->buf[i]);
  }
#endif

  if (offset >= 0) {
    offset = append_text(line, sizeof(line), offset, "\n");
  }
  if (offset > 0) {
    fwrite(line, 1, (size_t)offset, stdout);
  }
}

static esp_err_t csi_init(void) {
  wifi_csi_config_t csi_config = {
#if CONFIG_CSI_MODE_RAW_HTLTF
      .lltf_en = false,
      .htltf_en = true,
      .stbc_htltf2_en = false,
      .ltf_merge_en = false,
      .channel_filter_en = false,
      .manu_scale = false,
      .shift = 0,
#elif CONFIG_CSI_MODE_ROUTER_COMPATIBLE_LLTF
      .lltf_en = true,
      .htltf_en = false,
      .stbc_htltf2_en = false,
      .ltf_merge_en = true,
      .channel_filter_en = true,
      .manu_scale = true,
      .shift = 3,
#elif CONFIG_CSI_MODE_MERGED_STABLE
      .lltf_en = true,
      .htltf_en = true,
      .stbc_htltf2_en = false,
      .ltf_merge_en = true,
      .channel_filter_en = true,
      .manu_scale = true,
      .shift = 3,
#elif CONFIG_CSI_MODE_RESEARCH_FULL
      .lltf_en = true,
      .htltf_en = true,
      .stbc_htltf2_en = true,
      .ltf_merge_en = false,
      .channel_filter_en = false,
      .manu_scale = false,
      .shift = 0,
#endif
      .dump_ack_en = false,
  };

  ESP_RETURN_ON_ERROR(esp_wifi_set_csi_config(&csi_config), TAG,
                      "set CSI config failed");
  ESP_RETURN_ON_ERROR(esp_wifi_set_csi_rx_cb(csi_rx_callback, NULL), TAG,
                      "set CSI callback failed");
  ESP_RETURN_ON_ERROR(esp_wifi_set_csi(true), TAG, "enable CSI failed");
  ESP_LOGI(TAG, "CSI enabled, mode=%s, filters: ap=%s dst=%s ht=%s",
           csi_mode_name(), CONFIG_CSI_FILTER_AP_BSSID ? "on" : "off",
           CONFIG_CSI_FILTER_STA_DST_MAC ? "on" : "off",
           CONFIG_CSI_FILTER_HT_FRAMES ? "on" : "off");
  ESP_LOGI(TAG,
           "CSI_DATA,node_id,seq,local_time_us,rx_timestamp_us,src_mac,dst_mac,"
           "first_word_invalid,rx_seq,payload_len,rssi,channel,"
           "secondary_channel,rate,sig_mode,mcs,cwb,stbc,sgi,noise_floor,ant,"
           "sig_len,rx_state,csi_len,csi_bytes...");
  return ESP_OK;
}

static void ping_success_callback(esp_ping_handle_t hdl, void *args) {
  (void)hdl;
  (void)args;
  ping_success_count++;
}

static void ping_timeout_callback(esp_ping_handle_t hdl, void *args) {
  (void)hdl;
  (void)args;
  ping_timeout_count++;
}

static esp_err_t start_ap_gateway_ping(void) {
#if CONFIG_CSI_ENABLE_SELF_PING
  ESP_RETURN_ON_FALSE(wifi_ip_info_valid, ESP_ERR_INVALID_STATE, TAG,
                      "Wi-Fi IP info is not ready");

  esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();
  ip_addr_copy_from_ip4(ping_config.target_addr, wifi_ip_info.gw);
  ping_config.count = ESP_PING_COUNT_INFINITE;
  ping_config.interval_ms = 1000 / CONFIG_CSI_SELF_PING_RATE_HZ;
  if (ping_config.interval_ms == 0) {
    ping_config.interval_ms = 1;
  }
  ping_config.timeout_ms = CONFIG_CSI_SELF_PING_TIMEOUT_MS;
  ping_config.data_size = CONFIG_CSI_SELF_PING_PAYLOAD_BYTES;

  esp_ping_callbacks_t callbacks = {
      .cb_args = NULL,
      .on_ping_success = ping_success_callback,
      .on_ping_timeout = ping_timeout_callback,
      .on_ping_end = NULL,
  };

  ESP_RETURN_ON_ERROR(esp_ping_new_session(&ping_config, &callbacks,
                                           &gateway_ping),
                      TAG, "create ping session failed");
  ESP_RETURN_ON_ERROR(esp_ping_start(gateway_ping), TAG, "start ping failed");
  ESP_LOGI(TAG,
           "STA ping started: target=" IPSTR
           ", rate=%d Hz, interval=%" PRIu32 " ms, payload=%d bytes",
           IP2STR(&wifi_ip_info.gw), CONFIG_CSI_SELF_PING_RATE_HZ,
           ping_config.interval_ms, CONFIG_CSI_SELF_PING_PAYLOAD_BYTES);
#else
  ESP_LOGI(TAG, "STA self-ping disabled");
#endif
  return ESP_OK;
}

static void csi_stats_task(void *arg) {
  (void)arg;
  uint32_t last_printed = csi_printed_count;
  uint32_t last_ping_success = ping_success_count;
  uint32_t last_timeouts = ping_timeout_count;
  int64_t last_time_us = esp_timer_get_time();

  while (true) {
    vTaskDelay(pdMS_TO_TICKS(CONFIG_CSI_STATS_INTERVAL_SEC * 1000));

    const int64_t now_us = esp_timer_get_time();
    const uint32_t printed = csi_printed_count;
    const uint32_t ping_success = ping_success_count;
    const uint32_t timeouts = ping_timeout_count;
    const double elapsed_s = (double)(now_us - last_time_us) / 1000000.0;
    const double csi_fps =
        elapsed_s > 0.0 ? (double)(printed - last_printed) / elapsed_s : 0.0;
    const double ping_fps = elapsed_s > 0.0
                                ? (double)(ping_success - last_ping_success) /
                                      elapsed_s
                                : 0.0;

    ESP_LOGI(TAG,
             "CSI stats: fps=%.1f, rows=%" PRIu32 ", callbacks=%" PRIu32
             ", filtered=%" PRIu32
             " (ap=%" PRIu32 ", dst=%" PRIu32 ", ht=%" PRIu32 ")"
             ", null=%" PRIu32 ", rx_seq_gaps=%" PRIu32
             ", ping_ok=%.1f/s, ping_timeout_delta=%" PRIu32
             ", ping_timeouts=%" PRIu32,
             csi_fps, printed, csi_callback_count, csi_filtered_count,
             csi_filtered_ap_count, csi_filtered_dst_count,
             csi_filtered_sig_mode_count, csi_null_count, csi_rx_seq_gap_count,
             ping_fps, timeouts - last_timeouts, timeouts);

    last_printed = printed;
    last_ping_success = ping_success;
    last_timeouts = timeouts;
    last_time_us = now_us;
  }
}

void app_main(void) {
  init_nvs();
  ESP_LOGI(TAG, "ESP32-S3 role: STA CSI receiver, node=%d",
           CONFIG_CSI_NODE_ID);
  ESP_ERROR_CHECK(wifi_init_sta());
  ESP_ERROR_CHECK(csi_init());
  ESP_ERROR_CHECK(start_ap_gateway_ping());
  xTaskCreate(csi_stats_task, "csi_stats", 4096, NULL, 2, NULL);
}
