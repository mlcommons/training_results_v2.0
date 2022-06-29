// Copyright 2017 The Closure Rules Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package io.bazel.rules.closure.webfiles.server;

import com.google.common.net.HostAndPort;
import java.io.IOException;
import java.net.BindException;
import java.net.Inet6Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.ServerSocket;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.Enumeration;
import javax.inject.Inject;
import javax.net.ServerSocketFactory;

/** Utilities for networking. */
final class NetworkUtils {

  private static final int CONNECTION_BACKLOG = 5;
  private static final int MAX_BIND_ATTEMPTS = 10;

  private final ServerSocketFactory serverSocketFactory;

  @Inject
  NetworkUtils(ServerSocketFactory serverSocketFactory) {
    this.serverSocketFactory = serverSocketFactory;
  }

  /** Binds server socket, incrementing port on {@link BindException} if {@code shouldRetry}. */
  ServerSocket createServerSocket(HostAndPort bind, boolean tryAlternativePortsOnFailure)
      throws IOException {
    InetAddress host = InetAddress.getByName(bind.getHost());
    int port = bind.getPort();
    BindException bindException = null;
    for (int n = 0; n < MAX_BIND_ATTEMPTS; n++) {
      try {
        return serverSocketFactory.createServerSocket(port, CONNECTION_BACKLOG, host);
      } catch (BindException e) {
        if (port == 0 || !tryAlternativePortsOnFailure) {
          throw e;
        }
        if (bindException == null) {
          bindException = e;
        } else if (!e.equals(bindException)) {
          bindException.addSuppressed(e);
        }
        port++;
      }
    }
    throw bindException;
  }

  /** Turns {@code address} into a more human readable form. */
  static HostAndPort createUrlAddress(HostAndPort address) {
    if (address.getHost().equals("::") || address.getHost().equals("0.0.0.0")) {
      return address.getPortOrDefault(80) == 80
          ? HostAndPort.fromHost(getCanonicalHostName())
          : HostAndPort.fromParts(getCanonicalHostName(), address.getPort());
    } else {
      return address.getPortOrDefault(80) == 80 ? HostAndPort.fromHost(address.getHost()) : address;
    }
  }

  /**
   * Returns the fully-qualified domain name of the local host in all lower case.
   *
   * @throws RuntimeException to wrap {@link UnknownHostException} if the local host could not be
   *     resolved into an address
   */
  static String getCanonicalHostName() {
    try {
      return getExternalAddressOfLocalSystem().getCanonicalHostName().toLowerCase();
    } catch (UnknownHostException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Returns the externally-facing IPv4 network address of the local host.
   *
   * <p>This function implements a workaround for an
   * <a href="http://bugs.java.com/bugdatabase/view_bug.do?bug_id=4665037">issue</a> in
   * {@link InetAddress#getLocalHost}.
   *
   * <p><b>Note:</b> This code was pilfered from {@link "com.google.net.base.LocalHost"} which was
   * never made open source.
   *
   * @throws UnknownHostException if the local host could not be resolved into an address
   */
  private static InetAddress getExternalAddressOfLocalSystem() throws UnknownHostException {
    InetAddress localhost = InetAddress.getLocalHost();
    // If we have a loopback address, look for an address using the network cards.
    if (localhost.isLoopbackAddress()) {
      try {
        Enumeration<NetworkInterface> interfaces = NetworkInterface.getNetworkInterfaces();
        if (interfaces == null) {
          return localhost;
        }
        while (interfaces.hasMoreElements()) {
          NetworkInterface networkInterface = interfaces.nextElement();
          Enumeration<InetAddress> addresses = networkInterface.getInetAddresses();
          while (addresses.hasMoreElements()) {
            InetAddress address = addresses.nextElement();
            if (!(address.isLoopbackAddress()
                || address.isLinkLocalAddress()
                || address instanceof Inet6Address)) {
              return address;
            }
          }
        }
      } catch (SocketException e) {
        // Fall-through.
      }
    }
    return localhost;
  }
}
